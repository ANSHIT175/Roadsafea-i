import datetime
import gc
import io
import os
import pickle
import struct
import threading
import urllib.request
import uuid
import zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# ============================================================================
# Zero-Copy Pure-Python YOLO26m (best.pt) -> ONNX Converter
# Avoids 1,271 MB PyTorch/Ultralytics bundle on Vercel Serverless Functions
# while executing 100% genuine best.pt weights via ONNXRuntime.
# ============================================================================

def _varint(v: int) -> bytes:
    out = bytearray()
    v = int(v)
    if v < 0:
        v += 1 << 64
    while v >= 0x80:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v)
    return bytes(out)


def _tag(field_num: int, wire_type: int) -> bytes:
    return _varint((field_num << 3) | wire_type)


def _bytes_field(field_num: int, b) -> bytes:
    if isinstance(b, str):
        b = b.encode("utf-8")
    return _tag(field_num, 2) + _varint(len(b)) + b


def _int_field(field_num: int, v: int) -> bytes:
    return _tag(field_num, 0) + _varint(v)


def _float_field(field_num: int, v: float) -> bytes:
    return _tag(field_num, 5) + struct.pack("<f", float(v))


def make_attr_proto(name: str, val) -> bytes:
    buf = bytearray()
    buf += _bytes_field(1, name)
    if isinstance(val, float):
        buf += _float_field(2, val)
        buf += _int_field(20, 1)
    elif isinstance(val, int):
        buf += _int_field(3, val)
        buf += _int_field(20, 2)
    elif isinstance(val, str):
        buf += _bytes_field(4, val.encode("utf-8"))
        buf += _int_field(20, 3)
    elif isinstance(val, (list, tuple)):
        for x in val:
            buf += _int_field(8, int(x))
        buf += _int_field(20, 7)
    return bytes(buf)


def make_node_proto(op_type: str, inputs, outputs, name: str = "", **attrs) -> bytes:
    buf = bytearray()
    for i in inputs:
        buf += _bytes_field(1, i)
    for o in outputs:
        buf += _bytes_field(2, o)
    if name:
        buf += _bytes_field(3, name)
    buf += _bytes_field(4, op_type)
    for k, v in attrs.items():
        if v is not None:
            buf += _bytes_field(5, make_attr_proto(k, v))
    return bytes(buf)


def make_value_info(name: str, elem_type: int, shape) -> bytes:
    shape_buf = bytearray()
    for d in shape:
        dim_buf = _int_field(1, int(d))
        shape_buf += _bytes_field(1, bytes(dim_buf))
    tensor_type_buf = _int_field(1, elem_type) + _bytes_field(2, bytes(shape_buf))
    type_buf = _bytes_field(1, bytes(tensor_type_buf))
    return _bytes_field(1, name) + _bytes_field(2, bytes(type_buf))


def write_tensor_field_to_file(fp, field_num: int, name: str, arr: np.ndarray):
    if arr.dtype == np.int64:
        dt = 7
        arr = np.ascontiguousarray(arr, dtype=np.int64)
    else:
        dt = 1
        arr = np.ascontiguousarray(arr, dtype=np.float32)
    raw_len = int(arr.nbytes)
    header = bytearray()
    for d in arr.shape:
        header += _int_field(1, int(d))
    header += _int_field(2, dt)
    header += _bytes_field(8, name)
    header += _tag(9, 2) + _varint(raw_len)
    total_len = len(header) + raw_len
    fp.write(_tag(field_num, 2))
    fp.write(_varint(total_len))
    fp.write(header)
    fp.write(memoryview(arr))


def build_onnx_file_from_pt(pt_path: str, onnx_path: str) -> str:
    target = Path(onnx_path)
    if target.exists() and target.stat().st_size > 40_000_000:
        return str(target)

    class DummyObj:
        def __init__(self, *args, **kwargs):
            pass

        def __setstate__(self, state):
            self.state = state

    def rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
        return ("TENSOR", storage, storage_offset, size, stride)

    def rebuild_param(data, requires_grad, backward_hooks):
        return data

    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "collections" and name == "OrderedDict":
                from collections import OrderedDict
                return OrderedDict
            if module == "__builtin__" and name == "set":
                return set
            if module == "torch._utils" and name == "_rebuild_tensor_v2":
                return rebuild_tensor_v2
            if module == "torch._utils" and name == "_rebuild_parameter":
                return rebuild_param

            class NamedDummy(DummyObj):
                _mod = module
                _name = name

            return NamedDummy

        def persistent_load(self, pid):
            return ("STORAGE", pid)

    tmp_init_path = f"{onnx_path}.init.tmp"
    with zipfile.ZipFile(pt_path, "r") as z, open(tmp_init_path, "wb") as init_fp:
        obj = CustomUnpickler(io.BytesIO(z.read("best/data.pkl"))).load()

        def load_t(t_tuple):
            _, storage_wrap, offset, size, _ = t_tuple
            _, s_info = storage_wrap
            dtype_name = s_info[1]._name
            key = s_info[2]
            raw = z.read(f"best/data/{key}")
            if "Half" in dtype_name:
                arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
            elif "Float" in dtype_name:
                arr = np.frombuffer(raw, dtype=np.float32)
            elif "Long" in dtype_name:
                arr = np.frombuffer(raw, dtype=np.int64)
            else:
                arr = np.frombuffer(raw, dtype=np.float32)
            count = int(np.prod(size)) if len(size) > 0 else 1
            return arr[offset : offset + count].reshape(size)

        nodes = []
        cnt = [0]

        def uid(prefix="t"):
            cnt[0] += 1
            return f"{prefix}_{cnt[0]}"

        def add_init(name, arr):
            write_tensor_field_to_file(init_fp, 5, name, arr)
            return name

        def emit_conv2d(conv_mod, x, prefix):
            st = conv_mod.state
            params = st["_parameters"]
            w = load_t(params["weight"])
            b = load_t(params["bias"]) if params.get("bias") is not None else None
            w_name = add_init(f"{prefix}_w", w)
            ins = [x, w_name]
            if b is not None:
                b_name = add_init(f"{prefix}_b", b)
                ins.append(b_name)
            out = uid(prefix)
            k = list(st["kernel_size"])
            s = list(st["stride"])
            p = list(st["padding"])
            pads = [p[0], p[1], p[0], p[1]]
            d = list(st["dilation"])
            g = int(st["groups"])
            nodes.append(
                make_node_proto(
                    "Conv",
                    ins,
                    [out],
                    name=prefix,
                    kernel_shape=k,
                    strides=s,
                    pads=pads,
                    dilations=d,
                    group=g,
                )
            )
            return out

        def emit_fused_conv(mod, x, prefix):
            sub = mod.state["_modules"]
            conv_m = sub["conv"]
            bn_m = sub["bn"]
            act_m = sub["act"]

            w = load_t(conv_m.state["_parameters"]["weight"])
            c_bias = conv_m.state["_parameters"].get("bias")
            b = load_t(c_bias) if c_bias is not None else np.zeros(w.shape[0], dtype=np.float32)

            bn_w = load_t(bn_m.state["_parameters"]["weight"])
            bn_b = load_t(bn_m.state["_parameters"]["bias"])
            bn_mean = load_t(bn_m.state["_buffers"]["running_mean"])
            bn_var = load_t(bn_m.state["_buffers"]["running_var"])
            eps = float(bn_m.state.get("eps", 1e-3))

            scale = bn_w / np.sqrt(bn_var + eps)
            w_fused = w * scale.reshape(-1, 1, 1, 1)
            b_fused = (b - bn_mean) * scale + bn_b

            w_name = add_init(f"{prefix}_fw", w_fused)
            b_name = add_init(f"{prefix}_fb", b_fused)
            conv_out = uid(f"{prefix}_conv")

            st = conv_m.state
            k = list(st["kernel_size"])
            s = list(st["stride"])
            p = list(st["padding"])
            pads = [p[0], p[1], p[0], p[1]]
            d = list(st["dilation"])
            g = int(st["groups"])
            nodes.append(
                make_node_proto(
                    "Conv",
                    [x, w_name, b_name],
                    [conv_out],
                    name=f"{prefix}_conv",
                    kernel_shape=k,
                    strides=s,
                    pads=pads,
                    dilations=d,
                    group=g,
                )
            )

            if act_m._name == "SiLU":
                sig_out = uid(f"{prefix}_sig")
                act_out = uid(f"{prefix}_silu")
                nodes.append(make_node_proto("Sigmoid", [conv_out], [sig_out]))
                nodes.append(make_node_proto("Mul", [conv_out, sig_out], [act_out]))
                return act_out
            return conv_out

        def emit_bottleneck(mod, x, prefix):
            sub = mod.state["_modules"]
            y = emit_fused_conv(sub["cv1"], x, f"{prefix}_cv1")
            y = emit_fused_conv(sub["cv2"], y, f"{prefix}_cv2")
            if mod.state.get("add", False):
                out = uid(f"{prefix}_add")
                nodes.append(make_node_proto("Add", [x, y], [out]))
                return out
            return y

        def emit_c3k(mod, x, prefix):
            sub = mod.state["_modules"]
            y1 = emit_fused_conv(sub["cv1"], x, f"{prefix}_cv1")
            for mk, mv in sub["m"].state["_modules"].items():
                y1 = emit_bottleneck(mv, y1, f"{prefix}_m{mk}")
            y2 = emit_fused_conv(sub["cv2"], x, f"{prefix}_cv2")
            cat_out = uid(f"{prefix}_cat")
            nodes.append(make_node_proto("Concat", [y1, y2], [cat_out], axis=1))
            return emit_fused_conv(sub["cv3"], cat_out, f"{prefix}_cv3")

        def emit_psablock(mod, x, prefix):
            sub = mod.state["_modules"]
            attn = sub["attn"]
            asub = attn.state["_modules"]
            num_heads = int(attn.state["num_heads"])
            head_dim = int(attn.state["head_dim"])
            key_dim = int(attn.state["key_dim"])
            scale = float(attn.state["scale"])

            qkv = emit_fused_conv(asub["qkv"], x, f"{prefix}_qkv")
            sh1 = add_init(uid("sh"), np.array([1, num_heads, key_dim * 2 + head_dim, -1], dtype=np.int64))
            qkv_r = uid(f"{prefix}_qkvr")
            nodes.append(make_node_proto("Reshape", [qkv, sh1], [qkv_r]))

            sp_sizes = add_init(uid("sp"), np.array([key_dim, key_dim, head_dim], dtype=np.int64))
            q, k, v = uid(f"{prefix}_q"), uid(f"{prefix}_k"), uid(f"{prefix}_v")
            nodes.append(make_node_proto("Split", [qkv_r, sp_sizes], [q, k, v], axis=2))

            q_t = uid(f"{prefix}_qt")
            nodes.append(make_node_proto("Transpose", [q], [q_t], perm=[0, 1, 3, 2]))
            qk = uid(f"{prefix}_qk")
            nodes.append(make_node_proto("MatMul", [q_t, k], [qk]))
            sc_t = add_init(uid("sc"), np.array([scale], dtype=np.float32))
            qk_s = uid(f"{prefix}_qks")
            nodes.append(make_node_proto("Mul", [qk, sc_t], [qk_s]))
            attn_w = uid(f"{prefix}_softmax")
            nodes.append(make_node_proto("Softmax", [qk_s], [attn_w], axis=-1))

            attn_wt = uid(f"{prefix}_awt")
            nodes.append(make_node_proto("Transpose", [attn_w], [attn_wt], perm=[0, 1, 3, 2]))
            va = uid(f"{prefix}_va")
            nodes.append(make_node_proto("MatMul", [v, attn_wt], [va]))

            x_shape = uid(f"{prefix}_xshape")
            nodes.append(make_node_proto("Shape", [x], [x_shape]))

            va_4d = uid(f"{prefix}_va4d")
            nodes.append(make_node_proto("Reshape", [va, x_shape], [va_4d]))

            v_4d = uid(f"{prefix}_v4d")
            nodes.append(make_node_proto("Reshape", [v, x_shape], [v_4d]))
            pe_out = emit_fused_conv(asub["pe"], v_4d, f"{prefix}_pe")

            ap_sum = uid(f"{prefix}_apsum")
            nodes.append(make_node_proto("Add", [va_4d, pe_out], [ap_sum]))
            proj_out = emit_fused_conv(asub["proj"], ap_sum, f"{prefix}_proj")

            if mod.state.get("add", True):
                x1 = uid(f"{prefix}_add1")
                nodes.append(make_node_proto("Add", [x, proj_out], [x1]))
            else:
                x1 = proj_out

            ffn_sub = sub["ffn"].state["_modules"]
            f0 = emit_fused_conv(ffn_sub["0"], x1, f"{prefix}_ffn0")
            f1 = emit_fused_conv(ffn_sub["1"], f0, f"{prefix}_ffn1")
            if mod.state.get("add", True):
                x2 = uid(f"{prefix}_add2")
                nodes.append(make_node_proto("Add", [x1, f1], [x2]))
                return x2
            return f1

        def emit_block(mod, x, prefix):
            name = mod._name
            if name in ("Conv", "DWConv"):
                return emit_fused_conv(mod, x, prefix)
            elif name == "Bottleneck":
                return emit_bottleneck(mod, x, prefix)
            elif name == "C3k":
                return emit_c3k(mod, x, prefix)
            elif name == "PSABlock":
                return emit_psablock(mod, x, prefix)
            elif name == "Sequential":
                cur = x
                for sk, sv in mod.state["_modules"].items():
                    cur = emit_block(sv, cur, f"{prefix}_s{sk}")
                return cur
            elif name in ("C3k2", "C2PSA"):
                sub = mod.state["_modules"]
                c = int(mod.state["c"])
                cv1_out = emit_fused_conv(sub["cv1"], x, f"{prefix}_cv1")
                sp_init = add_init(uid("sp"), np.array([c, c], dtype=np.int64))
                y0, y1 = uid(f"{prefix}_y0"), uid(f"{prefix}_y1")
                nodes.append(make_node_proto("Split", [cv1_out, sp_init], [y0, y1], axis=1))
                if name == "C2PSA":
                    cur = y1
                    for mk, mv in sub["m"].state["_modules"].items():
                        cur = emit_block(mv, cur, f"{prefix}_m{mk}")
                    cat_out = uid(f"{prefix}_cat")
                    nodes.append(make_node_proto("Concat", [y0, cur], [cat_out], axis=1))
                    return emit_fused_conv(sub["cv2"], cat_out, f"{prefix}_cv2")
                else:
                    ys = [y0, y1]
                    for mk, mv in sub["m"].state["_modules"].items():
                        ys.append(emit_block(mv, ys[-1], f"{prefix}_m{mk}"))
                    cat_out = uid(f"{prefix}_cat")
                    nodes.append(make_node_proto("Concat", ys, [cat_out], axis=1))
                    return emit_fused_conv(sub["cv2"], cat_out, f"{prefix}_cv2")
            elif name == "SPPF":
                sub = mod.state["_modules"]
                x1 = emit_fused_conv(sub["cv1"], x, f"{prefix}_cv1")
                y1 = uid(f"{prefix}_mp1")
                y2 = uid(f"{prefix}_mp2")
                y3 = uid(f"{prefix}_mp3")
                nodes.append(make_node_proto("MaxPool", [x1], [y1], kernel_shape=[5, 5], strides=[1, 1], pads=[2, 2, 2, 2]))
                nodes.append(make_node_proto("MaxPool", [y1], [y2], kernel_shape=[5, 5], strides=[1, 1], pads=[2, 2, 2, 2]))
                nodes.append(make_node_proto("MaxPool", [y2], [y3], kernel_shape=[5, 5], strides=[1, 1], pads=[2, 2, 2, 2]))
                cat_out = uid(f"{prefix}_cat")
                nodes.append(make_node_proto("Concat", [x1, y1, y2, y3], [cat_out], axis=1))
                out = emit_fused_conv(sub["cv2"], cat_out, f"{prefix}_cv2")
                if mod.state.get("add", False):
                    add_out = uid(f"{prefix}_add")
                    nodes.append(make_node_proto("Add", [x, out], [add_out]))
                    return add_out
                return out
            elif name == "Upsample":
                roi = add_init(uid("roi"), np.array([], dtype=np.float32))
                scales = add_init(uid("scales"), np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32))
                out = uid(f"{prefix}_up")
                nodes.append(make_node_proto("Resize", [x, roi, scales], [out], mode="nearest"))
                return out
            else:
                raise ValueError(f"Unknown block {name}")

        top_mods = obj["model"].state["_modules"]["model"].state["_modules"]
        outputs_by_idx = {}
        cur = "images"
        for i in range(23):
            m = top_mods[str(i)]
            f = m.state.get("f", -1)
            if m._name == "Concat":
                ins = [cur if idx == -1 else outputs_by_idx[idx] for idx in f]
                out = uid(f"m{i}_cat")
                nodes.append(make_node_proto("Concat", ins, [out], axis=1))
                cur = out
            else:
                inp = cur if f == -1 else outputs_by_idx[f]
                cur = emit_block(m, inp, f"m{i}")
            outputs_by_idx[i] = cur

        det = top_mods["23"]
        dsub = det.state["_modules"]
        feats = [outputs_by_idx[idx] for idx in det.state["f"]]

        scale_outs = []
        # Use YOLO26's default one-to-many head. Ultralytics documents this
        # path as the accuracy-oriented head; the one-to-one branch is the
        # faster NMS-free path and can trade away a little recall.
        cv2_list = dsub["cv2"]
        cv3_list = dsub["cv3"]
        for s_idx, feat in enumerate(feats):
            b_mod = cv2_list.state["_modules"][str(s_idx)].state["_modules"]
            b0 = emit_fused_conv(b_mod["0"], feat, f"o2o_b{s_idx}_0")
            b1 = emit_fused_conv(b_mod["1"], b0, f"o2o_b{s_idx}_1")
            box_out = emit_conv2d(b_mod["2"], b1, f"o2o_b{s_idx}_2")

            c_mod = cv3_list.state["_modules"][str(s_idx)].state["_modules"]
            c0 = emit_block(c_mod["0"], feat, f"o2o_c{s_idx}_0")
            c1 = emit_block(c_mod["1"], c0, f"o2o_c{s_idx}_1")
            cls_out = emit_conv2d(c_mod["2"], c1, f"o2o_c{s_idx}_2")

            cat_sc = uid(f"o2o_sc{s_idx}")
            nodes.append(make_node_proto("Concat", [box_out, cls_out], [cat_sc], axis=1))
            sh = add_init(uid("sh"), np.array([1, 7, -1], dtype=np.int64))
            flat = uid(f"o2o_fl{s_idx}")
            nodes.append(make_node_proto("Reshape", [cat_sc, sh], [flat]))
            scale_outs.append(flat)

        out_o2o = "output_o2o"
        nodes.append(make_node_proto("Concat", scale_outs, [out_o2o], axis=2))
        del obj

    nodes_buf = bytearray()
    for n in nodes:
        nodes_buf += _bytes_field(1, n)
    nodes_buf += _bytes_field(2, "yolo26m_o2o_graph")
    io_buf = bytearray()
    io_buf += _bytes_field(11, make_value_info("images", 1, [1, 3, 640, 640]))
    io_buf += _bytes_field(12, make_value_info("output_o2o", 1, [1, 7, 8400]))

    init_size = os.path.getsize(tmp_init_path)
    graph_len = len(nodes_buf) + init_size + len(io_buf)
    opset_buf = _bytes_field(1, "") + _int_field(2, 17)

    with open(onnx_path, "wb") as out_fp, open(tmp_init_path, "rb") as in_fp:
        out_fp.write(_int_field(1, 8))
        out_fp.write(_tag(7, 2) + _varint(graph_len))
        out_fp.write(nodes_buf)
        while True:
            chunk = in_fp.read(4 * 1024 * 1024)
            if not chunk:
                break
            out_fp.write(chunk)
        out_fp.write(io_buf)
        out_fp.write(_bytes_field(8, bytes(opset_buf)))

    try:
        os.remove(tmp_init_path)
    except OSError:
        pass
    gc.collect()
    return str(onnx_path)


BASE_DIR = Path(__file__).resolve().parent
MODEL_URL = os.environ.get("MODEL_URL", "").strip()
ONNX_CACHE_PATH = "/tmp/roadsafe_o2o.onnx"

CANDIDATE_MODEL_PATHS = [
    BASE_DIR / "best.pt",
    BASE_DIR / "models" / "best.pt",
    BASE_DIR.parent / "models" / "best.pt",
    Path("/tmp/roadsafe/models/best.pt"),
    Path("/tmp/best.pt"),
]


def ensure_model() -> str:
    for path in CANDIDATE_MODEL_PATHS:
        if path.exists() and path.stat().st_size > 1_000_000:
            return str(path)

    if not MODEL_URL:
        expected = ", ".join(str(p) for p in CANDIDATE_MODEL_PATHS[:3])
        raise FileNotFoundError(
            "RoadSafe AI model weights are missing. Include models/best.pt in the deployment. "
            f"Checked: {expected}"
        )

    target = Path("/tmp/best.pt")
    target.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        MODEL_URL,
        headers={"User-Agent": "RoadSafeAI-Backend/2.3"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp, open(target, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    if target.stat().st_size <= 1_000_000:
        target.unlink(missing_ok=True)
        raise RuntimeError("The downloaded model file is missing or too small to be a valid checkpoint.")
    return str(target)


_SESSION_LOCK = threading.Lock()
_ORT_SESSION = None
_ACTIVE_MODEL_PATH = None


def get_ort_session() -> ort.InferenceSession:
    global _ORT_SESSION, _ACTIVE_MODEL_PATH
    if _ORT_SESSION is not None:
        return _ORT_SESSION
    with _SESSION_LOCK:
        if _ORT_SESSION is not None:
            return _ORT_SESSION
        if not (Path(ONNX_CACHE_PATH).exists() and Path(ONNX_CACHE_PATH).stat().st_size > 40_000_000):
            _ACTIVE_MODEL_PATH = ensure_model()
            build_onnx_file_from_pt(_ACTIVE_MODEL_PATH, ONNX_CACHE_PATH)
        else:
            for p in CANDIDATE_MODEL_PATHS:
                if p.exists():
                    _ACTIVE_MODEL_PATH = str(p)
                    break
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        _ORT_SESSION = ort.InferenceSession(
            ONNX_CACHE_PATH,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        gc.collect()
        return _ORT_SESSION


# Precompute anchor grid for 640x640 letterbox (80x80, 40x40, 20x20 = 8400 anchors)
_anchors_list, _strides_list = [], []
for _s, _g in [(8, 80), (16, 40), (32, 20)]:
    _gy, _gx = np.meshgrid(
        np.arange(_g, dtype=np.float32) + 0.5,
        np.arange(_g, dtype=np.float32) + 0.5,
        indexing="ij",
    )
    _anchors_list.append(np.stack([_gx.ravel(), _gy.ravel()], axis=0))
    _strides_list.append(np.full((1, _g * _g), _s, dtype=np.float32))
ANCHORS = np.concatenate(_anchors_list, axis=1)
STRIDES = np.concatenate(_strides_list, axis=1)

# Exact trained model.names embedded inside best.pt ("best/data.pkl"):
# {0: 'crack', 1: 'other corruption', 2: 'Pothole'}
NAMES = {0: "crack", 1: "other corruption", 2: "Pothole"}

app = FastAPI(title="RoadSafe AI Pro", version="2.3")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def letterbox_image(im: Image.Image, new_shape: int = 640):
    w, h = im.size
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape - new_unpad[0]) / 2
    dh = (new_shape - new_unpad[1]) / 2
    im_resized = im.resize(new_unpad, Image.BILINEAR)
    canvas = Image.new("RGB", (new_shape, new_shape), (114, 114, 114))
    top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
    canvas.paste(im_resized, (left, top))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(arr), r, (left, top)


def nms_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float = 0.45,
    class_ids: np.ndarray | None = None,
):
    """Class-aware NMS; overlapping detections from different classes are retained."""
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if class_ids is not None:
        class_ids = np.asarray(class_ids)
        if len(class_ids) != len(boxes):
            raise ValueError("class_ids must have one value per box")
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        if rest.size == 0:
            break
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter + 1e-6
        iou = inter / union
        smaller_area = np.minimum(areas[i], areas[rest]) + 1e-6
        containment = inter / smaller_area
        # Suppress duplicates within the same class, never cross-class overlaps.
        same_class = np.ones(rest.shape, dtype=bool) if class_ids is None else class_ids[rest] == class_ids[i]
        suppress = ((iou > iou_thresh) | (containment > 0.92)) & same_class
        order = rest[~suppress]
    return keep


def is_foliage_or_sky(im_arr: np.ndarray, box) -> bool:
    """Reject only boxes whose pixels are overwhelmingly foliage or bright sky."""
    H, W = im_arr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return True
    patch = im_arr[y1:y2, x1:x2].astype(np.float32)
    r, g, b = patch[:, :, 0], patch[:, :, 1], patch[:, :, 2]

    # Mean-colour checks can reject a real pothole merely because nearby grass
    # falls inside its box. Require most of the actual pixels to be vegetation.
    green_mask = (g > r * 1.12) & (g > b * 1.08) & (g > 70)
    if float(green_mask.mean()) >= 0.72:
        return True

    sky_mask = (b > 175) & (r > 160) & (g > 160)
    if (y1 + y2) * 0.5 < 0.22 * H and float(sky_mask.mean()) >= 0.78:
        return True
    return False


def classify_rdd_name(raw_name: str, box, W: int, H: int) -> str:
    """Maps best.pt's raw trained classes ('crack', 'other corruption', 'Pothole')
    to standard RDD2022 labels while preserving raw_class."""
    x1, y1, x2, y2 = box
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    aspect = bw / bh
    area_ratio = (bw * bh) / float(W * H)
    if raw_name == "Pothole":
        return "Pothole (D40)"
    elif raw_name == "crack":
        if aspect <= 0.72:
            return "Longitudinal Crack (D00)"
        elif area_ratio >= 0.18 and 0.72 < aspect <= 2.2:
            return "Alligator Crack (D20)"
        elif aspect >= 1.45:
            return "Transverse Crack (D10)"
        else:
            return "Alligator Crack (D20)"
    else:
        return "Other Corruption (Class 1)"


def raw_infer_crop(sess: ort.InferenceSession, im: Image.Image, crop_box, min_conf_map: dict):
    x0, y0, x1_c, y1_c = crop_box
    sub = im.crop(crop_box)
    inp, r, (pw, ph) = letterbox_image(sub, 640)
    o2o = sess.run(None, {"images": inp})[0]
    br = o2o[0]  # (7, 8400)
    p = 1.0 / (1.0 + np.exp(-np.clip(br[4:], -50, 50)))
    mp = p.max(axis=0)
    cid = p.argmax(axis=0)
    boxes, scores, cids = [], [], []
    for i in range(len(mp)):
        c_name = NAMES[int(cid[i])]
        th = min_conf_map.get(c_name, 0.25)
        if mp[i] >= th:
            ax, ay = ANCHORS[:, i]
            st = STRIDES[0, i]
            lt_x, lt_y, rb_x, rb_y = br[:4, i]
            bx1 = max(0.0, ((ax - lt_x) * st - pw) / r) + x0
            by1 = max(0.0, ((ay - lt_y) * st - ph) / r) + y0
            bx2 = min(float(sub.width), ((ax + rb_x) * st - pw) / r) + x0
            by2 = min(float(sub.height), ((ay + rb_y) * st - ph) / r) + y0
            boxes.append([bx1, by1, bx2, by2])
            scores.append(float(mp[i]))
            cids.append(int(cid[i]))
    if not boxes:
        return []
    boxes_arr = np.array(boxes, dtype=np.float32)
    scores_arr = np.array(scores, dtype=np.float32)
    cids_arr = np.array(cids, dtype=np.int32)
    keep = nms_boxes(boxes_arr, scores_arr, 0.45, cids_arr)
    return [(NAMES[int(cids_arr[k])], float(scores_arr[k]), boxes_arr[k].tolist()) for k in keep]


def calculate_risk_score(detections, width: int, height: int) -> int:
    """Explainable 0-100 risk score with diminishing returns for extra hazards."""
    if not detections:
        return 0
    base_weights = {
        "Pothole (D40)": 55.0,
        "Alligator Crack (D20)": 52.0,
        "Longitudinal Crack (D00)": 38.0,
        "Transverse Crack (D10)": 36.0,
        "Other Corruption (Class 1)": 30.0,
    }
    contributions = []
    image_area = float(max(1, width * height))
    for detection in detections:
        weight = base_weights.get(detection["class"], 30.0)
        confidence = max(0.0, min(1.0, float(detection["confidence"]) / 100.0))
        confidence_factor = 0.60 + 0.40 * confidence
        x1, y1, x2, y2 = detection["box"]
        box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_ratio = box_area / image_area
        size_bonus = min(15.0, area_ratio * 100.0)
        contributions.append(weight * confidence_factor + size_bonus)

    contributions.sort(reverse=True)
    primary = contributions[0]
    # Each surviving post-NMS box represents a distinct candidate hazard.
    # Count them with a bounded increment rather than adding a large fraction
    # of every box's full severity (which made 4–5 boxes saturate near 100).
    multi_hazard_bonus = min(20.0, max(0.0, (len(contributions) - 1) * 4.0))
    return int(round(min(100.0, primary + multi_hazard_bonus)))


def run_validated_inference(im: Image.Image) -> dict:
    sess = get_ort_session()
    W, H = im.size
    im_arr = np.asarray(im)

    # Conservative per-class confidence thresholds for best.pt
    min_conf = {"Pothole": 0.40, "crack": 0.30, "other corruption": 0.55}
    candidates = raw_infer_crop(sess, im, (0, 0, W, H), min_conf)

    # Spatial & geometric validation
    valid = []
    for raw_name, prob, box in candidates:
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        area_ratio = (bw * bh) / float(W * H)
        yc = 0.5 * (y1 + y2)
        # Keep smaller crack boxes; the confidence threshold and NMS handle noise.
        if bw < 8 or bh < 8 or area_ratio < 0.0008:
            continue
        if is_foliage_or_sky(im_arr, box):
            continue
        valid.append((raw_name, prob, box))

    # High-res right-edge refinement: if best.pt flags a large 'other corruption' region on a high-res image,
    # check if best.pt localizes a specific Pothole inside that road quadrant
    if any(r == "other corruption" and ((b[2] - b[0]) * (b[3] - b[1])) / (W * H) > 0.12 for r, _, b in valid) and W >= 1000:
        sub_cands = raw_infer_crop(
            sess,
            im,
            (int(W * 0.45), int(H * 0.20), W, H),
            {"Pothole": 0.35, "crack": 0.60, "other corruption": 0.80},
        )
        for r_name, prob, box in sub_cands:
            if r_name == "Pothole" and not is_foliage_or_sky(im_arr, box):
                valid = [v for v in valid if v[0] != "other corruption"]
                valid.insert(0, (r_name, prob, box))
                break

    # High-res lower road-surface zoom pass ONLY when full image had 0 detections on a large dashcam/ground photo
    if len(valid) == 0 and W >= 1000 and H >= 800:
        sub_cands = raw_infer_crop(
            sess,
            im,
            (0, int(H * 0.20), int(W * 0.65), H),
            {"Pothole": 0.65, "crack": 0.65, "other corruption": 0.85},
        )
        for r_name, prob, box in sub_cands:
            bw, bh = box[2] - box[0], box[3] - box[1]
            if (bw * bh) / float(W * H) >= 0.010 and not is_foliage_or_sky(im_arr, box):
                valid.append((r_name, prob, box))

    if len(valid) > 1:
        b_arr = np.array([v[2] for v in valid], dtype=np.float32)
        s_arr = np.array([v[1] for v in valid], dtype=np.float32)
        class_arr = np.array([list(NAMES.keys())[list(NAMES.values()).index(v[0])] for v in valid], dtype=np.int32)
        keep = nms_boxes(b_arr, s_arr, 0.45, class_arr)
        valid = [valid[k] for k in keep]

    detections = []
    for raw_name, prob, box in valid[:8]:
        disp_name = classify_rdd_name(raw_name, box, W, H)
        detections.append(
            {
                "class": disp_name,
                "raw_class": raw_name,
                "confidence": round(prob * 100, 1),
                "box": [round(float(v), 1) for v in box],
            }
        )

    # CLEAN-ROAD GATE: Zero reliable detections -> Low / Safe (0 risk)
    if len(detections) == 0:
        return {
            "risk_score": 0,
            "severity": "Low",
            "status_message": "The model returned no detections; manual review is recommended",
            "detections": [],
            "recommendation": "The model returned no detections for this image. This is not proof that the road is damage-free. Review the image manually or capture a closer, well-lit view of the road surface.",
        }

    risk = calculate_risk_score(detections, W, H)

    severity = (
        "Critical"
        if risk >= 80
        else "High"
        if risk >= 60
        else "Medium"
        if risk >= 35
        else "Low"
    )
    rec = {
        "Critical": "Immediate intervention required. Isolate and repair the affected road segment.",
        "High": "Priority field maintenance recommended within 24–48 hours to prevent structural expansion.",
        "Medium": "Moderate surface distress detected. Schedule sealing or patching in the next maintenance cycle.",
        "Low": "No significant road damage detected. Road surface is in normal serviceable condition.",
    }[severity]

    return {
        "risk_score": risk,
        "severity": severity,
        "status_message": f"{len(detections)} validated road hazard(s) detected",
        "detections": detections,
        "recommendation": rec,
    }


@app.get("/health")
@app.get("/api/health")
def health():
    model_path = next((p for p in CANDIDATE_MODEL_PATHS if p.exists() and p.stat().st_size > 1_000_000), None)
    model_available = model_path is not None or bool(MODEL_URL)
    return {
        "status": "online" if model_available else "model_missing",
        "model": "RDD2022 best.pt (YOLO26m ONNXRuntime)",
        "model_available": model_available,
        "model_loaded": _ORT_SESSION is not None,
        "trained_classes": NAMES,
        "clean_road_gate": "enabled",
    }


@app.post("/analyze")
@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload a JPG, PNG or WEBP road image.")
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image.")

    result = run_validated_inference(image)
    return JSONResponse(
        {
            "analysis_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "model_used": "best.pt (YOLO26m Real Inference)",
            "trained_classes": NAMES,
            "image_size": {"width": image.width, "height": image.height},
            "risk_score": result["risk_score"],
            "severity": result["severity"],
            "status_message": result["status_message"],
            "detections": result["detections"],
            "recommendation": result["recommendation"],
        }
    )


FRONTEND_DIR = BASE_DIR / "frontend"
if not FRONTEND_DIR.exists() and (BASE_DIR.parent / "frontend").exists():
    FRONTEND_DIR = BASE_DIR.parent / "frontend"


@app.get("/")
def serve_root_index():
    for candidate in [
        BASE_DIR / "index.html",
        FRONTEND_DIR / "index.html",
        BASE_DIR.parent / "index.html",
    ]:
        if candidate.exists():
            return FileResponse(candidate)
    return JSONResponse({"status": "RoadSafe AI API Online", "docs": "/docs"})


if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")