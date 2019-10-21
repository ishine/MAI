def if_android(a):
    return select({
        "//:android": a,
        "//conditions:default": [],
    })

def if_neon_enabled(a):
    return select({
        "//:neon_enabled": a,
        "//conditions:default": [],
    })

def if_tensorflow_enabled(a):
    return select({
        "//:tensorflow_enabled": a,
        "//conditions:default": [],
    })

def if_onnx_enabled(a):
    return select({
        "//:onnx_enabled": a,
        "//conditions:default": [],
    })

