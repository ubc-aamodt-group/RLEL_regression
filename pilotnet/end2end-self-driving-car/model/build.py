from model import meta_arch


def build_model(cfg, visualizing=False,init="u"):
    print(init)
    model_factory = getattr(meta_arch, cfg.MODEL.META_ARCHITECTURE)
    model = model_factory(cfg, visualizing,init=init)
    return model


def build_backward_model(cfg):
    model_factory = getattr(meta_arch, cfg.MODEL.BACKWARD_META_ARCHITECTURE)
    backward_model = model_factory(cfg)
    return backward_model
