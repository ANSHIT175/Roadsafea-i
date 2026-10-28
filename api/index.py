import zipfile, pickle, io, struct, time
import numpy as np
import onnxruntime as ort
from PIL import Image

# Minimal pure-Python Protobuf encoder for ONNX (ModelProto / GraphProto / NodeProto / TensorProto)
def _varint(v):
    out = bytearray()
    v = int(v)
    if v < 0:
        v += (1 << 64)
    while v >= 0x80:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v)
    return bytes(out)

def _tag(field_num, wire_type):
    return _varint((field_num << 3) | wire_type)

def _bytes_field(field_num, b):
    if isinstance(b, str):
        b = b.encode('utf-8')
    return _tag(field_num, 2) + _varint(len(b)) + b

def _int_field(field_num, v):
    return _tag(field_num, 0) + _varint(v)

def _float_field(field_num, v):
    return _tag(field_num, 5) + struct.pack('<f', float(v))

def make_tensor_proto(name, arr):
    # TensorProto: dims=1(rep int64), data_type=2(int32: 1=FLOAT, 7=INT64), name=8(string), raw_data=9(bytes)
    if arr.dtype == np.int64:
        dt = 7
        raw = arr.tobytes()
    else:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        dt = 1
        raw = arr.tobytes()
    buf = bytearray()
    for d in arr.shape:
        buf += _int_field(1, int(d))
    buf += _int_field(2, dt)
    buf += _bytes_field(8, name)
    buf += _bytes_field(9, raw)
    return bytes(buf)

def make_attr_proto(name, val):
    # AttributeProto: name=1, f=2(float), i=3(int), s=4(bytes), t=5(TensorProto), ints=8(rep int64), type=20
    # Types: FLOAT=1, INT=2, STRING=3, TENSOR=4, FLOATS=6, INTS=7
    buf = bytearray()
    buf += _bytes_field(1, name)
    if isinstance(val, float):
        buf += _float_field(2, val)
        buf += _int_field(20, 1)
    elif isinstance(val, int):
        buf += _int_field(3, val)
        buf += _int_field(20, 2)
    elif isinstance(val, str):
        buf += _bytes_field(4, val.encode('utf-8'))
        buf += _int_field(20, 3)
    elif isinstance(val, (list, tuple)):
        for x in val:
            buf += _int_field(8, int(x))
        buf += _int_field(20, 7)
    return bytes(buf)

def make_node_proto(op_type, inputs, outputs, name="", **attrs):
    # NodeProto: input=1, output=2, name=3, op_type=4, attribute=5
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

def make_value_info(name, elem_type, shape):
    # ValueInfoProto: name=1, type=2 (TypeProto: tensor_type=1 (TypeProto.Tensor: elem_type=1, shape=2 (TensorShapeProto: dim=1 (Dimension: dim_value=1))))
    shape_buf = bytearray()
    for d in shape:
        dim_buf = _int_field(1, int(d))
        shape_buf += _bytes_field(1, bytes(dim_buf))
    tensor_type_buf = _int_field(1, elem_type) + _bytes_field(2, bytes(shape_buf))
    type_buf = _bytes_field(1, bytes(tensor_type_buf))
    return _bytes_field(1, name) + _bytes_field(2, bytes(type_buf))

def build_onnx_from_pt(pt_path):
    class DummyObj:
        def __init__(self, *args, **kwargs): pass
        def __setstate__(self, state): self.state = state

    def rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
        return ('TENSOR', storage, storage_offset, size, stride)

    def rebuild_param(data, requires_grad, backward_hooks):
        return data

    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'collections' and name == 'OrderedDict':
                from collections import OrderedDict
                return OrderedDict
            if module == '__builtin__' and name == 'set': return set
            if module == 'torch._utils' and name == '_rebuild_tensor_v2': return rebuild_tensor_v2
            if module == 'torch._utils' and name == '_rebuild_parameter': return rebuild_param
            class NamedDummy(DummyObj):
                _mod = module; _name = name
            return NamedDummy
        def persistent_load(self, pid): return ('STORAGE', pid)

    with zipfile.ZipFile(pt_path, 'r') as z:
        data = z.read('best/data.pkl')
        obj = CustomUnpickler(io.BytesIO(data)).load()
        
        def load_t(t_tuple):
            _, storage_wrap, offset, size, stride = t_tuple
            _, s_info = storage_wrap
            dtype_name = s_info[1]._name
            key = s_info[2]
            raw = z.read(f'best/data/{key}')
            if 'Half' in dtype_name:
                arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
            elif 'Float' in dtype_name:
                arr = np.frombuffer(raw, dtype=np.float32)
            elif 'Long' in dtype_name:
                arr = np.frombuffer(raw, dtype=np.int64)
            else:
                arr = np.frombuffer(raw, dtype=np.float32)
            count = int(np.prod(size)) if len(size) > 0 else 1
            return arr[offset:offset+count].reshape(size)

        nodes = []
        initializers = []
        cnt = [0]
        def uid(prefix="t"):
            cnt[0] += 1
            return f"{prefix}_{cnt[0]}"

        def add_init(name, arr):
            initializers.append(make_tensor_proto(name, arr))
            return name

        def emit_conv2d(conv_mod, x, prefix):
            st = conv_mod.state
            params = st['_parameters']
            w = load_t(params['weight'])
            b = load_t(params['bias']) if params.get('bias') is not None else None
            w_name = add_init(f"{prefix}_w", w)
            ins = [x, w_name]
            if b is not None:
                b_name = add_init(f"{prefix}_b", b)
                ins.append(b_name)
            out = uid(prefix)
            k = list(st['kernel_size'])
            s = list(st['stride'])
            p = list(st['padding'])
            pads = [p[0], p[1], p[0], p[1]]
            d = list(st['dilation'])
            g = int(st['groups'])
            nodes.append(make_node_proto("Conv", ins, [out], name=prefix, kernel_shape=k, strides=s, pads=pads, dilations=d, group=g))
            return out

        def emit_fused_conv(mod, x, prefix):
            # mod is ultralytics Conv or DWConv: has conv, bn, act
            sub = mod.state['_modules']
            conv_m = sub['conv']
            bn_m = sub['bn']
            act_m = sub['act']
            
            w = load_t(conv_m.state['_parameters']['weight'])
            c_bias = conv_m.state['_parameters'].get('bias')
            b = load_t(c_bias) if c_bias is not None else np.zeros(w.shape[0], dtype=np.float32)
            
            bn_w = load_t(bn_m.state['_parameters']['weight'])
            bn_b = load_t(bn_m.state['_parameters']['bias'])
            bn_mean = load_t(bn_m.state['_buffers']['running_mean'])
            bn_var = load_t(bn_m.state['_buffers']['running_var'])
            eps = float(bn_m.state.get('eps', 1e-3))
            
            scale = bn_w / np.sqrt(bn_var + eps)
            w_fused = w * scale.reshape(-1, 1, 1, 1)
            b_fused = (b - bn_mean) * scale + bn_b
            
            w_name = add_init(f"{prefix}_fw", w_fused)
            b_name = add_init(f"{prefix}_fb", b_fused)
            conv_out = uid(f"{prefix}_conv")
            
            st = conv_m.state
            k = list(st['kernel_size'])
            s = list(st['stride'])
            p = list(st['padding'])
            pads = [p[0], p[1], p[0], p[1]]
            d = list(st['dilation'])
            g = int(st['groups'])
            nodes.append(make_node_proto("Conv", [x, w_name, b_name], [conv_out], name=f"{prefix}_conv", kernel_shape=k, strides=s, pads=pads, dilations=d, group=g))
            
            if act_m._name == 'SiLU':
                sig_out = uid(f"{prefix}_sig")
                act_out = uid(f"{prefix}_silu")
                nodes.append(make_node_proto("Sigmoid", [conv_out], [sig_out]))
                nodes.append(make_node_proto("Mul", [conv_out, sig_out], [act_out]))
                return act_out
            return conv_out

        def emit_bottleneck(mod, x, prefix):
            sub = mod.state['_modules']
            y = emit_fused_conv(sub['cv1'], x, f"{prefix}_cv1")
            y = emit_fused_conv(sub['cv2'], y, f"{prefix}_cv2")
            if mod.state.get('add', False):
                out = uid(f"{prefix}_add")
                nodes.append(make_node_proto("Add", [x, y], [out]))
                return out
            return y

        def emit_c3k(mod, x, prefix):
            sub = mod.state['_modules']
            y1 = emit_fused_conv(sub['cv1'], x, f"{prefix}_cv1")
            for mk, mv in sub['m'].state['_modules'].items():
                y1 = emit_bottleneck(mv, y1, f"{prefix}_m{mk}")
            y2 = emit_fused_conv(sub['cv2'], x, f"{prefix}_cv2")
            cat_out = uid(f"{prefix}_cat")
            nodes.append(make_node_proto("Concat", [y1, y2], [cat_out], axis=1))
            return emit_fused_conv(sub['cv3'], cat_out, f"{prefix}_cv3")

        def emit_psablock(mod, x, prefix):
            sub = mod.state['_modules']
            attn = sub['attn']
            asub = attn.state['_modules']
            num_heads = int(attn.state['num_heads']) # 4
            head_dim = int(attn.state['head_dim'])   # 64
            key_dim = int(attn.state['key_dim'])     # 32
            scale = float(attn.state['scale'])
            
            # qkv: (1, 512, H, W) where H=20, W=20 (N=400)
            qkv = emit_fused_conv(asub['qkv'], x, f"{prefix}_qkv")
            # Reshape to (1, num_heads, key_dim*2 + head_dim, -1) -> (1, 4, 128, 400)
            sh1 = add_init(uid("sh"), np.array([1, num_heads, key_dim * 2 + head_dim, -1], dtype=np.int64))
            qkv_r = uid(f"{prefix}_qkvr")
            nodes.append(make_node_proto("Reshape", [qkv, sh1], [qkv_r]))
            
            # Split along axis=2 into [32, 32, 64]
            sp_sizes = add_init(uid("sp"), np.array([key_dim, key_dim, head_dim], dtype=np.int64))
            q, k, v = uid(f"{prefix}_q"), uid(f"{prefix}_k"), uid(f"{prefix}_v")
            nodes.append(make_node_proto("Split", [qkv_r, sp_sizes], [q, k, v], axis=2))
            
            # attn = (q.transpose(-2, -1) @ k) * scale
            # q_t: (1, 4, 400, 32), k: (1, 4, 32, 400) -> matmul: (1, 4, 400, 400)
            q_t = uid(f"{prefix}_qt")
            nodes.append(make_node_proto("Transpose", [q], [q_t], perm=[0, 1, 3, 2]))
            qk = uid(f"{prefix}_qk")
            nodes.append(make_node_proto("MatMul", [q_t, k], [qk]))
            sc_t = add_init(uid("sc"), np.array([scale], dtype=np.float32))
            qk_s = uid(f"{prefix}_qks")
            nodes.append(make_node_proto("Mul", [qk, sc_t], [qk_s]))
            attn_w = uid(f"{prefix}_softmax")
            nodes.append(make_node_proto("Softmax", [qk_s], [attn_w], axis=-1))
            
            # x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
            attn_wt = uid(f"{prefix}_awt")
            nodes.append(make_node_proto("Transpose", [attn_w], [attn_wt], perm=[0, 1, 3, 2]))
            va = uid(f"{prefix}_va")
            nodes.append(make_node_proto("MatMul", [v, attn_wt], [va]))
            
            # Get dynamic shape of x or use Shape(x) so it works for any H, W!
            x_shape = uid(f"{prefix}_xshape")
            nodes.append(make_node_proto("Shape", [x], [x_shape]))
            
            va_4d = uid(f"{prefix}_va4d")
            nodes.append(make_node_proto("Reshape", [va, x_shape], [va_4d]))
            
            v_4d = uid(f"{prefix}_v4d")
            nodes.append(make_node_proto("Reshape", [v, x_shape], [v_4d]))
            pe_out = emit_fused_conv(asub['pe'], v_4d, f"{prefix}_pe")
            
            ap_sum = uid(f"{prefix}_apsum")
            nodes.append(make_node_proto("Add", [va_4d, pe_out], [ap_sum]))
            proj_out = emit_fused_conv(asub['proj'], ap_sum, f"{prefix}_proj")
            
            if mod.state.get('add', True):
                x1 = uid(f"{prefix}_add1")
                nodes.append(make_node_proto("Add", [x, proj_out], [x1]))
            else:
                x1 = proj_out
                
            ffn_sub = sub['ffn'].state['_modules']
            f0 = emit_fused_conv(ffn_sub['0'], x1, f"{prefix}_ffn0")
            f1 = emit_fused_conv(ffn_sub['1'], f0, f"{prefix}_ffn1")
            if mod.state.get('add', True):
                x2 = uid(f"{prefix}_add2")
                nodes.append(make_node_proto("Add", [x1, f1], [x2]))
                return x2
            return f1

        def emit_block(mod, x, prefix):
            name = mod._name
            if name == 'Conv' or name == 'DWConv':
                return emit_fused_conv(mod, x, prefix)
            elif name == 'Bottleneck':
                return emit_bottleneck(mod, x, prefix)
            elif name == 'C3k':
                return emit_c3k(mod, x, prefix)
            elif name == 'PSABlock':
                return emit_psablock(mod, x, prefix)
            elif name == 'Sequential':
                cur = x
                for sk, sv in mod.state['_modules'].items():
                    cur = emit_block(sv, cur, f"{prefix}_s{sk}")
                return cur
            elif name in ('C3k2', 'C2PSA'):
                sub = mod.state['_modules']
                c = int(mod.state['c'])
                cv1_out = emit_fused_conv(sub['cv1'], x, f"{prefix}_cv1")
                sp_init = add_init(uid("sp"), np.array([c, c], dtype=np.int64))
                y0, y1 = uid(f"{prefix}_y0"), uid(f"{prefix}_y1")
                nodes.append(make_node_proto("Split", [cv1_out, sp_init], [y0, y1], axis=1))
                if name == 'C2PSA':
                    cur = y1
                    for mk, mv in sub['m'].state['_modules'].items():
                        cur = emit_block(mv, cur, f"{prefix}_m{mk}")
                    cat_out = uid(f"{prefix}_cat")
                    nodes.append(make_node_proto("Concat", [y0, cur], [cat_out], axis=1))
                    return emit_fused_conv(sub['cv2'], cat_out, f"{prefix}_cv2")
                else:
                    ys = [y0, y1]
                    for mk, mv in sub['m'].state['_modules'].items():
                        ys.append(emit_block(mv, ys[-1], f"{prefix}_m{mk}"))
                    cat_out = uid(f"{prefix}_cat")
                    nodes.append(make_node_proto("Concat", ys, [cat_out], axis=1))
                    return emit_fused_conv(sub['cv2'], cat_out, f"{prefix}_cv2")
            elif name == 'SPPF':
                sub = mod.state['_modules']
                x1 = emit_fused_conv(sub['cv1'], x, f"{prefix}_cv1")
                y1 = uid(f"{prefix}_mp1")
                y2 = uid(f"{prefix}_mp2")
                y3 = uid(f"{prefix}_mp3")
                nodes.append(make_node_proto("MaxPool", [x1], [y1], kernel_shape=[5, 5], strides=[1, 1], pads=[2, 2, 2, 2]))
                nodes.append(make_node_proto("MaxPool", [y1], [y2], kernel_shape=[5, 5], strides=[1, 1], pads=[2, 2, 2, 2]))
                nodes.append(make_node_proto("MaxPool", [y2], [y3], kernel_shape=[5, 5], strides=[1, 1], pads=[2, 2, 2, 2]))
                cat_out = uid(f"{prefix}_cat")
                nodes.append(make_node_proto("Concat", [x1, y1, y2, y3], [cat_out], axis=1))
                out = emit_fused_conv(sub['cv2'], cat_out, f"{prefix}_cv2")
                if mod.state.get("add", False):
                    add_out = uid(f"{prefix}_add")
                    nodes.append(make_node_proto("Add", [x, out], [add_out]))
                    return add_out
                return out
            elif name == 'Upsample':
                roi = add_init(uid("roi"), np.array([], dtype=np.float32))
                scales = add_init(uid("scales"), np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32))
                out = uid(f"{prefix}_up")
                nodes.append(make_node_proto("Resize", [x, roi, scales], [out], mode="nearest"))
                return out
            else:
                raise ValueError(f"Unknown block {name}")

        top_mods = obj['model'].state['_modules']['model'].state['_modules']
        outputs_by_idx = {}
        cur = "images"
        for i in range(23):
            m = top_mods[str(i)]
            f = m.state.get('f', -1)
            if m._name == 'Concat':
                ins = [cur if idx == -1 else outputs_by_idx[idx] for idx in f]
                out = uid(f"m{i}_cat")
                nodes.append(make_node_proto("Concat", ins, [out], axis=1))
                cur = out
            else:
                inp = cur if f == -1 else outputs_by_idx[f]
                cur = emit_block(m, inp, f"m{i}")
            outputs_by_idx[i] = cur

        # Detect head (module 23)
        det = top_mods['23']
        dsub = det.state['_modules']
        feats = [outputs_by_idx[idx] for idx in det.state['f']] # [16, 19, 22]
        
        # Emit both one2one (end2end) and one2many (cv2/cv3) outputs so we can inspect both!
        def emit_head_branch(cv2_list, cv3_list, tag_prefix):
            scale_outs = []
            for s_idx, feat in enumerate(feats):
                b_mod = cv2_list.state['_modules'][str(s_idx)].state['_modules']
                b0 = emit_fused_conv(b_mod['0'], feat, f"{tag_prefix}_b{s_idx}_0")
                b1 = emit_fused_conv(b_mod['1'], b0, f"{tag_prefix}_b{s_idx}_1")
                box_out = emit_conv2d(b_mod['2'], b1, f"{tag_prefix}_b{s_idx}_2")
                
                c_mod = cv3_list.state['_modules'][str(s_idx)].state['_modules']
                c0 = emit_block(c_mod['0'], feat, f"{tag_prefix}_c{s_idx}_0")
                c1 = emit_block(c_mod['1'], c0, f"{tag_prefix}_c{s_idx}_1")
                cls_out = emit_conv2d(c_mod['2'], c1, f"{tag_prefix}_c{s_idx}_2")
                
                cat_sc = uid(f"{tag_prefix}_sc{s_idx}")
                nodes.append(make_node_proto("Concat", [box_out, cls_out], [cat_sc], axis=1))
                sh = add_init(uid("sh"), np.array([1, 7, -1], dtype=np.int64))
                flat = uid(f"{tag_prefix}_fl{s_idx}")
                nodes.append(make_node_proto("Reshape", [cat_sc, sh], [flat]))
                scale_outs.append(flat)
            out_name = f"output_{tag_prefix}"
            nodes.append(make_node_proto("Concat", scale_outs, [out_name], axis=2))
            return out_name

        out_o2o = emit_head_branch(dsub['one2one_cv2'], dsub['one2one_cv3'], "o2o")
        out_o2m = emit_head_branch(dsub['cv2'], dsub['cv3'], "o2m")

        # Assemble GraphProto (node=1, name=2, initializer=5, input=11, output=12)
        graph_buf = bytearray()
        for n in nodes:
            graph_buf += _bytes_field(1, n)
        graph_buf += _bytes_field(2, "yolo26m_graph")
        for init in initializers:
            graph_buf += _bytes_field(5, init)
        graph_buf += _bytes_field(11, make_value_info("images", 1, [1, 3, 640, 640]))
        graph_buf += _bytes_field(12, make_value_info(out_o2o, 1, [1, 7, 8400]))
        graph_buf += _bytes_field(12, make_value_info(out_o2m, 1, [1, 7, 8400]))

        # Assemble ModelProto (ir_version=1(int64=8), graph=7, opset_import=8(OperatorSetIdProto: domain=1, version=2(17)))
        opset_buf = _bytes_field(1, "") + _int_field(2, 17)
        model_buf = _int_field(1, 8) + _bytes_field(7, bytes(graph_buf)) + _bytes_field(8, bytes(opset_buf))
        return bytes(model_buf)



import datetime
import os
import urllib.request
import uuid
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://github.com/ANSHIT175/RoadSafei/releases/download/v1.0/best.pt",
)

CANDIDATE_MODEL_PATHS = [
    BASE_DIR / "best.pt",
    BASE_DIR / "models" / "best.pt",
    BASE_DIR.parent / "models" / "best.pt",
    Path("/tmp/best.pt"),
]


def ensure_model() -> str:
    for path in CANDIDATE_MODEL_PATHS:
        if path.exists() and path.stat().st_size > 1_000_000:
            return str(path)
    target = Path("/tmp/best.pt")
    target.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        MODEL_URL,
        headers={"User-Agent": "RoadSafeAI-Backend/2.0"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp, open(target, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return str(target)


MODEL_PATH = ensure_model()
ONNX_BYTES = build_onnx_from_pt(MODEL_PATH)
ORT_SESSION = ort.InferenceSession(ONNX_BYTES, providers=["CPUExecutionProvider"])

# Precompute anchor grid for 640x640 letterbox (80x80, 40x40, 20x20)
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

NAMES = {0: "crack", 1: "other corruption", 2: "Pothole"}
ALIASES = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole",
    "longitudinal crack": "Longitudinal Crack",
    "transverse crack": "Transverse Crack",
    "alligator crack": "Alligator Crack",
    "pothole": "Pothole",
    "crack": "Road Crack",
    "other corruption": "Surface Corruption",
}
WEIGHTS = {
    "Pothole": 38,
    "Alligator Crack": 30,
    "Road Crack": 25,
    "Crack": 25,
    "Longitudinal Crack": 22,
    "Transverse Crack": 22,
    "Surface Corruption": 20,
    "Other Corruption": 20,
}

app = FastAPI(title="RoadSafe AI Pro", version="2.0")
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


@app.get("/health")
@app.get("/api/health")
def health():
    return {
        "status": "online",
        "model": "RDD2022 best.pt (YOLO26m ONNXRuntime)",
        "model_path": MODEL_PATH,
        "classes": NAMES,
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

    inp, r, (pad_w, pad_h) = letterbox_image(image, 640)
    o2o, _ = ORT_SESSION.run(None, {"images": inp})
    out = o2o[0]  # (7, 8400)
    box_raw = out[:4, :]
    cls_prob = 1.0 / (1.0 + np.exp(-np.clip(out[4:, :], -50, 50)))
    max_prob = cls_prob.max(axis=0)
    cls_id = cls_prob.argmax(axis=0)

    mask = max_prob >= 0.25
    idxs = np.where(mask)[0]
    idxs = idxs[np.argsort(-max_prob[idxs])][:10]

    detections = []
    for idx in idxs:
        ax, ay = ANCHORS[:, idx]
        st = STRIDES[0, idx]
        lt_x, lt_y, rb_x, rb_y = box_raw[:, idx]
        x1 = max(0.0, ((ax - lt_x) * st - pad_w) / r)
        y1 = max(0.0, ((ay - lt_y) * st - pad_h) / r)
        x2 = min(float(image.width), ((ax + rb_x) * st - pad_w) / r)
        y2 = min(float(image.height), ((ay + rb_y) * st - pad_h) / r)
        raw_name = NAMES.get(int(cls_id[idx]), "crack")
        name = ALIASES.get(raw_name, ALIASES.get(raw_name.lower(), raw_name))
        conf = round(float(max_prob[idx]) * 100, 1)
        detections.append(
            {
                "class": name,
                "raw_class": raw_name,
                "confidence": conf,
                "box": [round(float(v), 1) for v in (x1, y1, x2, y2)],
            }
        )

    risk = min(
        100,
        int(
            round(
                sum(
                    WEIGHTS.get(d["class"], 20) * (d["confidence"] / 100)
                    for d in detections
                )
                + min(25, len(detections) * 5)
            )
        ),
    )
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
        "Critical": "Immediate intervention. Inspect and isolate the affected road segment.",
        "High": "Priority field inspection recommended within 24 hours.",
        "Medium": "Schedule maintenance inspection in the next cycle.",
        "Low": "Monitor and verify during routine inspection.",
    }[severity]

    return JSONResponse(
        {
            "analysis_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "image_size": {"width": image.width, "height": image.height},
            "risk_score": risk,
            "severity": severity,
            "detections": detections,
            "recommendation": rec,
        }
    )


FRONTEND_DIR = BASE_DIR / "frontend"
if not FRONTEND_DIR.exists() and (BASE_DIR.parent / "frontend").exists():
    FRONTEND_DIR = BASE_DIR.parent / "frontend"

if (BASE_DIR / "index.html").exists():
    @app.get("/")
    def serve_root_index():
        return FileResponse(BASE_DIR / "index.html")
elif FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
