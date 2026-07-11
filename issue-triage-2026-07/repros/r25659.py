# Issue 25659 gave no standalone repro; it reports that tree_util's prefix-error
# message builder uses str(k.key), which crashes with AttributeError for
# GetAttrKey (which has .name, not .key). Construct a pytree whose child keys
# are GetAttrKey and trigger a prefix error with differing child counts.
import jax
from jax.tree_util import register_pytree_with_keys, GetAttrKey
from jax._src.tree_util import prefix_errors


class Obj:
    def __init__(self, **kw):
        self.d = kw


def flatten_with_keys(o):
    return [(GetAttrKey(k), v) for k, v in o.d.items()], tuple(o.d.keys())


def flatten(o):
    return list(o.d.values()), tuple(o.d.keys())


def unflatten(keys, vals):
    return Obj(**dict(zip(keys, vals)))


register_pytree_with_keys(Obj, flatten_with_keys, unflatten, flatten)

prefix = Obj(a=1)
full = Obj(a=1, b=2)

errs = prefix_errors(prefix, full)
print("got", len(errs), "prefix errors")
e = errs[0]("my_tree")  # building the error message is what crashed
print("error message built successfully:")
print(e)
