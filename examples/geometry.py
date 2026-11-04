import marimo

__generated_with = '0.25.2'
app = marimo.App(width='medium')


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import quantities as pq
    from geometry_viewer import GeometryViewer

    from pyframetree import CoordinateSystem

    mo.output.clear()  # Hide fastplotlib's optional Jupyter logo/device banner.
    return CoordinateSystem, GeometryViewer, mo, np, pq


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # See how a parent frame moves its child

    The **Reference** frame stays fixed. **Main** is the parent of **Child**,
    which carries a point cloud. Axis labels identify each frame and its local
    **u**, **v**, and **w** directions. Labels stay upright as the camera moves.
    All positions are in metres.

    Drag the 3D view to rotate it; scroll to zoom. Change the angles below to
    update the geometry without resetting your camera. The defaults reproduce
    the original Jupyter example.

    Open this notebook with `uv run --group notebook marimo edit examples/geometry.py`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    parent_angle = mo.ui.slider(
        -180,
        180,
        value=30,
        step=5,
        show_value=True,
        label='Parent rotation (deg)',
    )
    child_angle = mo.ui.slider(
        -180,
        180,
        value=30,
        step=5,
        show_value=True,
        label='Child rotation (deg)',
    )
    parent_scale = mo.ui.slider(
        1,
        4,
        value=3,
        step=0.25,
        show_value=True,
        label='Parent axis scale',
    )
    inherit_scale = mo.ui.checkbox(value=False, label='Scale the child with its parent')
    mo.vstack([parent_angle, child_angle, parent_scale, inherit_scale])
    return child_angle, inherit_scale, parent_angle, parent_scale


@app.cell
def _(np):
    # Keep the same cloud when a control changes, so only the geometry moves.
    points = np.random.default_rng(0).random((100, 3))
    return (points,)


@app.cell
def _(CoordinateSystem, child_angle, inherit_scale, parent_angle, parent_scale, points, pq):
    reference = CoordinateSystem(name='Reference')
    main = CoordinateSystem(name='Main')
    child = CoordinateSystem(parent=main, name='Child')
    child.add_points(points, 'cloud')
    child.translate([2, 2, 0] * pq.m)
    child.rotate([0, 1, 1], child_angle.value * pq.deg)

    main.scale(parent_scale.value, inherit=inherit_scale.value)
    _ = main.rotate([1, 0, 0], parent_angle.value * pq.deg)
    return child, main, reference


@app.cell
def _(GeometryViewer, points):
    viewer = GeometryViewer(points)
    viewer.canvas
    return (viewer,)


@app.cell
def _(child, main, reference, viewer):
    viewer.update(reference, main, child)
    return


@app.cell(hide_code=True)
def _(mo):
    reset_view = mo.ui.run_button(label='Reset view')
    reset_view
    return (reset_view,)


@app.cell
def _(reset_view, viewer):
    if reset_view.value:
        viewer.reset_view()
    return


@app.cell(hide_code=True)
def _(child, main, mo, pq, reference):
    _rows = []
    for _frame in (reference, main, child):
        _position = _frame.origin.rescale(pq.m).magnitude
        _rows.append(
            {
                'Frame': _frame.name,
                'Parent': _frame.parent.name if _frame.parent else '—',
                'x (m)': round(float(_position[0]), 4),
                'y (m)': round(float(_position[1]), 4),
                'z (m)': round(float(_position[2]), 4),
            }
        )
    mo.vstack(
        [
            mo.md('### World positions'),
            mo.ui.table(_rows, selection=None, pagination=False, show_column_summaries=False),
            mo.md("""
        The parent rotates about the world x-axis. The child rotates about its
        own origin, around its parent's axis `(0, 1, 1)`.

        With **Scale the child with its parent** unchecked, `inherit=False`
        changes the parent's axes while holding the child and its points fixed.
        Rotation keeps the default `inherit=True`, so the child follows its parent.
        """),
        ]
    )
    return


if __name__ == '__main__':
    app.run()
