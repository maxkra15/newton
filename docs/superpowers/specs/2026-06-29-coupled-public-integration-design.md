# Coupled-Solver Public Integration Design

## Goal

Make Newton's experimental coupled solvers safe to integrate into vectorized
simulation frameworks without wrappers around sub-solvers or access to Newton
private state. The work targets masked reset isolation, entry-local collision,
contact reporting, graph-capture discovery, coupling diagnostics, and batched
VBD joint-mode configuration.

The implementation is based on Newton PR #2848 commit
`c948e5116ab40d1543858e2c99e9fe2726364925` and remains within the existing
`newton.solvers.experimental.coupled` API boundary.

## Design principles

- Preserve the existing full-reset and stepping behavior by default.
- Keep the real sub-solver object visible to the coupler so its
  `CouplingInterface` overrides cannot be hidden by decorators.
- Keep contacts in one declared namespace and expose mappings whenever a
  compact entry view uses local indices.
- Do not add simulation overhead for diagnostics unless diagnostics are
  requested.
- Prefer existing public extension points over parallel configuration APIs.
- Treat a world mask as a semantic isolation guarantee, not merely a hint.

## 1. Masked coupled reset

`SolverCoupled.reset(state, world_mask=..., flags=...)` already advertises a
world-selective contract. The base solver forwards the mask to entries, but
base transient clearing and the Proxy and ADMM implementations currently clear
or rewrite whole arrays.

The existing signature will remain unchanged. The implementation will:

1. Validate that a supplied mask is a boolean Warp array on the model device
   with `model.world_count` elements.
2. Copy and clear entry-state fields only for entities whose `*_world` value is
   selected. Static entities with world `-1` are changed only by a full reset.
3. Preserve unselected Proxy feedback, previous-force, velocity-snapshot, and
   relaxation state. Aitken accumulators and relaxation values will become
   per-world so one environment cannot affect another environment's iteration
   history.
4. Preserve unselected ADMM primal, dual, proximal, matching, and warm-start
   rows. Each resettable row will carry or derive a world id from its body,
   particle, joint, or contact endpoints.
5. Force contact detection to refresh after a partial reset when stale geometry
   may remain. A global collision pass is acceptable when the collision
   pipeline cannot update a subset, but it must not clear unselected coupling
   history.
6. Keep `world_mask=None` as the optimized full-array reset path.

Masked-reset tests will initialize two worlds with distinct nonzero histories,
reset one world, and assert bitwise preservation of the other world's state for
base, fixed/Aitken Proxy, and ADMM coupling.

## 2. Entry-local collision pipelines

`SolverCoupled.Entry` will gain two optional fields:

```python
collision_pipeline: Callable[[ModelView], CollisionPipeline | None] | None = None
collide_interval: int | None = None
```

The factory is called after the entry `ModelView` is finalized and before the
entry solver is constructed. Returning `None` selects the existing outer
contact path. A non-`None` pipeline owns a persistent `Contacts` buffer and is
refreshed after parent state distribution, once per outer coupled step according
to `collide_interval`.

Contact precedence is explicit:

1. Contacts supplied by a coupling algorithm for a particular solve, such as
   Proxy destination contacts or an ADMM internal interface, take precedence.
2. Otherwise an entry-local pipeline is used when configured.
3. Otherwise outer contacts are filtered through the existing entry view.

This lets MuJoCo use local Newton collision while remaining the actual
`SolverBase`/`CouplingInterface` instance. It eliminates the need for a wrapper
that can mask articulated effective-mass or gravity hooks. Existing Proxy-level
`collision_pipeline` remains supported because it describes a coupling mapping,
not the entry's ordinary contact source.

Construction validates callable factories, positive intervals, and the
pipeline's `contacts()`/`collide()` protocol. Tests will cover hook identity,
cadence, outer fallback, algorithm precedence, and CUDA capture preparation.

## 3. Public coupled contact streams

The experimental coupled package will expose an immutable
`CoupledContactStream` descriptor containing:

- a stable stream name;
- kind: `outer`, `entry`, `proxy`, or `admm`;
- the `Contacts` buffer;
- optional source and destination entry names;
- local-to-parent shape and particle maps when the buffer is entry-local;
- whether solved force data is available;
- current counts, capacities, and overflow values on explicit diagnostic
  request.

Stream names are `outer`, `entry/<name>`,
`proxy/<source>/<destination>`, and `admm/internal`.

`SolverCoupled.contact_streams(outer_contacts=None)` will return the streams
known after the most recent step. Base entries contribute local or filtered
streams, Proxy contributes each internal collision direction, and ADMM
contributes its internal detected-contact stream. Stream enumeration does not
copy contact buffers or synchronize the device.

`CoupledContactStream.diagnostics()` performs the optional device-to-host read
and reports raw count, capacity, and `max(count - capacity, 0)` overflow for
rigid and soft contacts. Force availability is explicit because ADMM's raw
collision buffer does not necessarily contain reconstructed solver forces.
Newton will not implicitly merge streams in `SolverCoupled.update_contacts()`;
an application must select streams deliberately to avoid duplicate contacts or
ambiguous force semantics.

## 4. Proxy view isolation

No second proxy-specific view callback will be added. The existing
`SolverCoupled.Entry.configure_view` callback already runs on the destination
`ModelView` after proxy entities are made visible and before solver
construction. It is therefore the canonical place for proxy-only material,
margin, and shape-property overrides.

Documentation and tests will demonstrate configuring proxy destination shapes
through copy-on-write and verify that the parent model and source entry retain
their original material arrays. Shape selection uses the public
`ModelView.shape_body` array; no second callback or selection helper is added.

## 5. CUDA graph capability and preparation

`SolverBase` will define two additive public hooks:

```python
@property
def supports_cuda_graph_capture(self) -> bool: ...

def prepare_cuda_graph_capture(self, contacts: Contacts | None = None) -> None: ...
```

The defaults are `True` and a no-op. Solvers with dynamic allocation or dynamic
topology override the property. Solvers that require persistent scratch buffers
allocate them in the preparation hook without advancing simulation state.

`SolverCoupled` reports support only when all entries and all configured local
collision providers support capture. A collision provider may expose
`supports_cuda_graph_capture` and `prepare_cuda_graph_capture()`; an absent
capability flag means supported and an absent preparation method is a no-op,
matching the base solver defaults. The coupled preparation hook forwards to
every entry, prepares local/proxy/ADMM contact buffers, and retains
`prepare_contacts()` as a compatible specialized helper. Tests will verify
nested capability rejection and that preparation performs no integration or
state mutation.

## 6. Public diagnostics and VBD controls

Proxy will provide read-only accessors for body and particle feedback keyed by
`(source, destination)`. Each result includes parent-model entity ids and the
corresponding force or spatial-wrench array; callers no longer need
`_proxy_mappings`. A separate on-demand diagnostics call reports fixed/Aitken
relaxation state per world.

ADMM will retain its existing contact-count properties and add an on-demand
diagnostics snapshot containing configured iterations, active/current and
high-water contact counts, and per-interface primal and dual residual norms.
Residual reductions are launched only when the snapshot is requested.

`SolverVBD.set_joint_constraint_modes(joint_indices, hard, slot=None)` will
batch the existing runtime operation. It accepts one mode for all selected
joints or a mode sequence of equal length, validates every joint before
mutation, updates structural slots in one transaction, and clears affected
soft-mode lambda/C0 history. The existing singular method delegates to the
batch method and remains compatible.

## Error handling and compatibility

All new APIs are additive. Existing constructors and full-reset behavior remain
valid. Invalid world masks, unknown stream or proxy directions, mismatched
batch lengths, and unsupported graph configurations fail before launching
kernels. Public docs and examples will import only from public Newton modules,
never `newton._src`.

Because the coupled API is experimental, the additions stay in its current
experimental namespace. User-visible additions and reset fixes receive entries
in the `[Unreleased]` sections of `CHANGELOG.md`.

## Verification

Implementation follows red-green TDD. The focused suite will include:

- two-world Base, Proxy, and ADMM reset-isolation tests;
- entry-local collision and `CouplingInterface` dispatch tests;
- contact-stream namespace, mapping, force-semantics, and overflow tests;
- proxy destination copy-on-write isolation tests;
- graph support aggregation and preparation-without-step tests;
- Proxy/ADMM diagnostic accessor tests;
- scalar-versus-batched VBD joint-mode parity and validation tests.

The final gate is the full coupled suite, relevant VBD and collision suites,
`uvx pre-commit run -a`, API documentation generation for new public symbols,
and a review of the exact diff against the PR #2848 base.

## Out of scope

IsaacLab selector resolution, callback registration, configuration unions,
legacy manager migration, Kamino configuration conversion, RTX step accounting,
and benchmark methodology remain IsaacLab work. Newton will provide the public
primitives those fixes consume, but will not acquire IsaacLab-specific scene or
sensor abstractions.
