
def import_class(name: str):
    """dynamically import the desired class as a variable"""
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def count_params(model):
    """Return the total number of trainable parameters of a given model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)