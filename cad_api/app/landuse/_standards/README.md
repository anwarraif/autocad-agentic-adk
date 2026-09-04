# Shared layer → land use standards

A file here maps layer names to land uses for **many drawings at once**, not
for one drawing. Its shape is exactly the same as a per-drawing override in
`landuse/_configs/`, so a mapping that has proved correct on one drawing can be
moved here without being rewritten.

A drawing uses one by naming it:

```yaml
# landuse/_configs/<drawing_id>.yaml
standard: <name-of-a-file-in-here>
```

The order, strongest first: **per-drawing override → shared standard → layer
name pattern**. A drawing can always overrule its standard.

## What must not be in here

**Guessed content.** This directory is deliberately empty. The layer codes in a
drawing are made by the engineer who drew it, and their meaning is in the
legend key that engineer wrote — often as an image pasted into the file itself
(see `/drawings/{id}/embedded`), not inside the layer names. Writing a standard
from a reading of layer name patterns means promoting one drawing's guess into
a rule for every drawing, and then marking it `verified` — exactly the mistake
this layer was built to prevent.

The content of a standard file is only valid if it comes from one of:

- a legend key read from the drawing (an embedded payload, or text);
- a layer naming standard document from its owner;
- a human decision recorded in `sources`.

The `sources` inside it must not name the geometry of any drawing: the size of
a plot is a fact about one file, whereas what travels to another file is only
the mapping.
