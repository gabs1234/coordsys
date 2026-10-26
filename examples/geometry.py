import marimo

__generated_with = '0.25.2'
app = marimo.App(width='medium')


@app.cell
def _():
    import json

    import marimo as mo
    import numpy as np
    import pyvista as pv
    import quantities as pq
    from trame_pyvista.jupyter import launch_server
    from trame_pyvista.widgets import get_server

    from coordsys import CoordinateSystem

    return CoordinateSystem, get_server, json, launch_server, mo, np, pq, pv


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # See how a parent frame moves its child

    The **Reference** frame stays fixed. **Main** is the parent of **Child**,
    which carries a point cloud. Axis labels identify each frame and its local
    **u**, **v**, and **w** directions. Labels stay upright as the camera moves.
    All positions are in metres.

    Drag the 3D view to rotate it; scroll to zoom. Change the angles below to
    rebuild the scene. The defaults reproduce the original Jupyter example.

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
    reference = CoordinateSystem()
    main = CoordinateSystem()
    child = main.add_child(label='Child')
    child.add_points(points, label='cloud', space='local')
    child.translate([2, 2, 0] * pq.m)
    child.rotate_euler_local([0, 1, 1], child_angle.value * pq.deg)

    main.scale(parent_scale.value, inherit=inherit_scale.value)
    _ = main.rotate_euler_local([1, 0, 0], parent_angle.value * pq.deg, inherit=True)
    return child, main, reference


@app.cell
def _(json):
    def with_axis_labels(html, labels, positions):
        """Project world anchors into a transparent overlay in the exported viewer."""
        anchors = json.dumps(
            [{'text': text, 'point': point.tolist()} for text, point in zip(labels, positions)]
        ).replace('<', '\\u003c')
        # PyVista's VTK.js export exposes its renderWindow. Using screen-space
        # text keeps labels upright and a constant size through orbit and zoom.
        script = r"""
        <script>
        (() => {
            const anchors = AXIS_LABELS;
            function attachLabels() {
                const rw = window.global?.renderWindow || window.renderWindow;
                const renderer = rw?.getRenderers().find(r => r.getActors().length);
                if (!renderer) {
                    requestAnimationFrame(attachLabels);
                    return;
                }
                const view = rw.getViews()[0];
                const layer = document.createElement('div');
                layer.style.cssText =
                    'position:fixed;inset:0;pointer-events:none;overflow:hidden;z-index:1';
                const nodes = anchors.map(({text}) => {
                    const label = document.createElement('span');
                    label.className = 'axis-label';
                    label.textContent = text;
                    label.style.cssText =
                        'position:absolute;font:13px sans-serif;color:#222;white-space:nowrap;' +
                        'transform:translate(-50%,-50%);text-shadow:0 0 3px white,0 0 3px white';
                    layer.appendChild(label);
                    return label;
                });
                document.body.appendChild(layer);
                function update() {
                    const bounds = view.getCanvas().getBoundingClientRect();
                    const [width, height] = view.getSize();
                    anchors.forEach(({point}, index) => {
                        const [x, y, z] = view.worldToDisplay(...point, renderer);
                        const label = nodes[index];
                        label.style.left = `${bounds.left + x * bounds.width / width}px`;
                        label.style.top = `${bounds.top + (1 - y / height) * bounds.height}px`;
                        label.style.display = z >= 0 && z <= 1 ? '' : 'none';
                    });
                }
                renderer.getActiveCamera().onModified(update);
                view.onModified(update);
                new ResizeObserver(update).observe(view.getCanvas());
                requestAnimationFrame(update);
            }
            requestAnimationFrame(attachLabels);
        })();
        </script>
        """.replace('AXIS_LABELS', anchors)
        return html.replace('</body>', script + '</body>')

    return (with_axis_labels,)


@app.cell
async def _(child, get_server, launch_server, main, np, pq, pv, reference, with_axis_labels):
    # Embed a self-contained viewer instead of opening a desktop plotting window.
    _server = get_server(pv.global_theme.trame.jupyter_server_name)
    # Marimo executes cells in a worker thread, where OS signal handlers are unavailable.
    _server.cli.set_defaults(nosignalhandlers=True)
    launch_server(_server)
    _plotter = None
    try:
        await _server.ready
        _plotter = pv.Plotter(off_screen=True, window_size=(900, 600))
        _plotter.set_background('white')
        _labels, _label_positions = [], []
        for _name, _frame, _color in [
            ('Reference', reference, '#666666'),
            ('Main', main, '#4e79a7'),
            ('Child', child, '#b86b16'),
        ]:
            _origin = _frame.origin.rescale(pq.m).magnitude
            for _axis_name, _axis in zip(('u', 'v', 'w'), (_frame.u, _frame.v, _frame.w)):
                _plotter.add_arrows(_origin, _axis, color=_color)
                _labels.append(f'{_name}.{_axis_name}')
                _label_positions.append(_origin + 1.12 * _axis)
        _plotter.add_points(child.get_points('cloud'), color='#b86b16', point_size=7)
        _plotter.add_lines(
            np.array([main.origin.magnitude, child.origin.magnitude]),
            color='#666666',
            width=1,
        )
        _plotter.view_isometric()
        _plotter.enable_parallel_projection()
        _plotter.reset_camera()
        _plotter.camera.zoom(0.8)  # Leave room for labels when the view is rotated.
        scene_html = with_axis_labels(
            _plotter.trame.export_html(None).getvalue(),
            _labels,
            _label_positions,
        )
    finally:
        if _plotter is not None:
            _plotter.close()
        _ = await _server.stop()
    return (scene_html,)


@app.cell(hide_code=True)
def _(mo, scene_html):
    mo.iframe(scene_html, height='600px')
    return


@app.cell(hide_code=True)
def _(child, main, mo, pq, reference):
    _rows = []
    for _name, _frame in [('Reference', reference), ('Main', main), ('Child', child)]:
        _position = _frame.origin.rescale(pq.m).magnitude
        _rows.append(
            {
                'Frame': _name,
                'Parent': 'Main' if _frame.get_parent() is main else '—',
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
        own origin, around the world axis `(0, 1, 1)`.

        With **Scale the child with its parent** unchecked, `inherit=False`
        changes the parent's axes while holding the child and its points fixed.
        Rotation still uses `inherit=True`, so the child follows its parent.
        """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, scene_html):
    mo.download(
        scene_html.encode('utf-8'),
        filename='coordsys.html',
        label='Download the interactive geometry',
    )
    return


if __name__ == '__main__':
    app.run()
