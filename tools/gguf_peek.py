import struct, sys, re

def rd(f, fmt):
    n = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(n))

def rstr(f):
    (n,) = rd(f, "<Q")
    return f.read(n).decode("utf-8", "replace")

# GGUF value types
def rval(f, t):
    if t == 0:  return rd(f, "<B")[0]
    if t == 1:  return rd(f, "<b")[0]
    if t == 2:  return rd(f, "<H")[0]
    if t == 3:  return rd(f, "<h")[0]
    if t == 4:  return rd(f, "<I")[0]
    if t == 5:  return rd(f, "<i")[0]
    if t == 6:  return rd(f, "<f")[0]
    if t == 7:  return rd(f, "<?")[0]
    if t == 8:  return rstr(f)
    if t == 9:
        (et,) = rd(f, "<I"); (n,) = rd(f, "<Q")
        return [rval(f, et) for _ in range(n)]
    if t == 10: return rd(f, "<Q")[0]
    if t == 11: return rd(f, "<q")[0]
    if t == 12: return rd(f, "<d")[0]
    raise ValueError("unknown type %d" % t)

path = sys.argv[1]
with open(path, "rb") as f:
    magic = f.read(4)
    assert magic == b"GGUF", magic
    ver, = rd(f, "<I")
    ntensor, = rd(f, "<Q")
    nkv, = rd(f, "<Q")
    kv = {}
    for _ in range(nkv):
        k = rstr(f)
        t, = rd(f, "<I")
        v = rval(f, t)
        kv[k] = v
    names = []
    for _ in range(ntensor):
        nm = rstr(f)
        nd, = rd(f, "<I")
        dims = [rd(f, "<Q")[0] for _ in range(nd)]
        tt, = rd(f, "<I")
        off, = rd(f, "<Q")
        names.append((nm, dims))

print("=== %s" % path)
print("gguf v%d, %d tensors, %d kv" % (ver, ntensor, nkv))
for k in sorted(kv):
    if any(s in k.lower() for s in ("mtp", "nextn", "spec", "draft", "arch", "block_count", "name")):
        v = kv[k]
        if isinstance(v, list) and len(v) > 8: v = "[list len %d]" % len(v)
        print("  KV %s = %r" % (k, v))
pat = re.compile(r"mtp|nextn|next_n|eh_proj|draft|shared_head", re.I)
hits = [(n, d) for n, d in names if pat.search(n)]
print("MTP-verdaechtige Tensoren: %d" % len(hits))
for n, d in hits:
    print("   ", n, d)
# highest block index
blk = sorted({int(m.group(1)) for n,_ in names if (m := re.match(r"blk\.(\d+)\.", n))})
print("blk-Indizes: %d..%d (%d Stueck)" % (blk[0], blk[-1], len(blk)))
# show all tensors of the last two blocks
for b in blk[-2:]:
    tn = [n for n,_ in names if n.startswith("blk.%d." % b)]
    print("  blk.%d: %d Tensoren" % (b, len(tn)))
top = [n for n,_ in names if not n.startswith("blk.")]
print("Nicht-blk-Tensoren:", top)
