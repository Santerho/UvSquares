#    <Uv Squares, Blender addon for reshaping UV vertices to grid.>
#    Copyright (C) <2020> <Reslav Hollos>
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program. If not, see <http://www.gnu.org/licenses/>.

bl_info = {
    "name": "UV Squares",
    "description": "UV Editor tool for reshaping quad selection to grid.",
    "author": "Reslav Hollos (Updated by <Your Name>)",
    "version": (1, 16, 0),
    "blender": (4, 3, 2),  # or at least (2, 80, 0) if you want older versions to work as well
    "location": "UV Editor > N Panel > UV Squares",
    "category": "UV",
    "wiki_url": "https://blendermarket.com/products/uv-squares"
}

import bpy
import bmesh
from collections import defaultdict
from math import hypot
from timeit import default_timer as timer

precision = 3

def main(context, operator, square=False, snapToClosest=False):
    """
    The main routine that checks context and calls the relevant logic
    for shaping lines or faces (rectangles / squares).
    """
    # Make sure "Keep UV and edit mesh in sync" is off
    if context.scene.tool_settings.use_uv_select_sync:
        operator.report({'ERROR'}, "Please disable 'Keep UV and edit mesh in sync'.")
        return

    selected_objects = list(context.selected_objects)
    if context.edit_object not in selected_objects:
        selected_objects.append(context.edit_object)

    for obj in selected_objects:
        if obj.type == "MESH":
            main_per_object(obj, context, operator, square, snapToClosest)

def main_per_object(obj, context, operator, square, snapToClosest):
    """
    Per-object main logic: gather UV data, figure out if the user
    is reshaping a line or a rectangular face, etc.
    """
    if context.scene.tool_settings.use_uv_select_sync:
        operator.report({'ERROR'}, "Please disable 'Keep UV and edit mesh in sync'")
        return

    startTime = timer()

    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    uv_layer = bm.loops.layers.uv.verify()

    # Gather selection sets
    edgeVerts, filteredVerts, selFaces, nonQuadFaces, vertsDict, noEdge = get_selection_lists(
        uv_layer, bm
    )

    # if nothing or only one vertex is selected:
    if len(filteredVerts) == 0:
        return
    if len(filteredVerts) == 1:
        # you can snap 2D cursor to that single vertex if you like
        snap_cursor_to_closest_selected([filteredVerts[0]])
        return

    # If no faces are selected, we are effectively dealing with
    # a line of UVs or arbitrary points:
    if len(selFaces) == 0:
        if snapToClosest:
            snap_cursor_to_closest_selected(filteredVerts)
            return
        # Fill the dictionary for these selected vertices:
        fill_verts_dict_line(uv_layer, bm, filteredVerts, vertsDict)

        # If they are already strictly along X or Y, distribute them equally.
        # Otherwise, first snap them to one axis, then call again to distribute.
        if not are_verts_aligned_on_x_or_y(filteredVerts):
            scale_to_axis_zero(filteredVerts, vertsDict, cursor_closest_to(filteredVerts))
            finish_success(me, startTime)
            return
        else:
            make_equal_distance_line(filteredVerts, vertsDict, cursor_closest_to(filteredVerts))
            finish_success(me, startTime)
            return

    # If we do have faces selected, we might want to drop any non-quads
    for nf in nonQuadFaces:
        for l in nf.loops:
            luv = l[uv_layer]
            luv.select = False

    # helper method to check if a face is fully selected
    def is_face_selected(f):
        if not f.select:
            return False
        return all(l[uv_layer].select for l in f.loops)

    # figure out face islands
    def get_island_from_face(start_face):
        island = set()
        to_check = {start_face}
        while to_check:
            f = to_check.pop()
            if is_face_selected(f) and f not in island:
                island.add(f)
                # walk across edges that are NOT seams
                for e in f.edges:
                    if not e.seam:
                        for linked_face in e.link_faces:
                            if linked_face != f:
                                to_check.add(linked_face)
        return island

    def get_islands(faces):
        all_islands = []
        remaining = set(faces)
        while remaining:
            f = remaining.pop()
            isl = get_island_from_face(f)
            all_islands.append(isl)
            remaining.difference_update(isl)
        return all_islands

    islands = get_islands(selFaces)

    # For each island, pick an active face to 'shape' as the reference
    def shape_island(target_face, faces, square_mode):
        shape_face(uv_layer, operator, target_face)
        if square_mode:
            follow_active_uv(operator, me, target_face, faces, 'EVEN')
        else:
            follow_active_uv(operator, me, target_face, faces)

    for island in islands:
        tf = bm.faces.active
        # If there's no valid active face or it’s not in this island,
        # or if we have multiple islands, pick any face in the island
        # that is a quad.
        if (
            tf is None
            or tf not in island
            or len(islands) > 1
            or not tf.select
            or len(tf.verts) != 4
        ):
            # fallback
            tf = next(iter(island))

        shape_island(tf, island, square)

    # If we had to “rip” the edge to separate quads from non-quads, re-join them
    if not noEdge:
        # edge has been “ripped” so we connect it back
        for ev in edgeVerts:
            key = (round(ev.uv.x, precision), round(ev.uv.y, precision))
            if key in vertsDict:
                ev.uv = vertsDict[key][0].uv
                ev.select = True

    finish_success(me, startTime)

# -----------------------------------------------------------------------------
#  Core geometry logic
# -----------------------------------------------------------------------------

def get_selection_lists(uv_layer, bm):
    """
    Collects various sets:
    - edgeVerts: selected loops (fully selected edges / faces, etc.)
    - filteredVerts: de-duplicated selected UVs
    - selFaces: fully selected faces that are quads
    - nonQuadFaces: fully selected faces that are not quads
    - vertsDict: dictionary that maps (x,y) to list of loops
    - noEdge: a bool indicating if edge was present or not
    """
    edgeVerts = []
    allEdgeVerts = []
    filteredVerts = []
    selFaces = []
    nonQuadFaces = []
    vertsDict = defaultdict(list)

    # Gather fully selected faces vs partially selected (edge) loops
    for f in bm.faces:
        if not f.select:
            continue
        face_edge_verts = []
        is_face_fully_sel = True
        for l in f.loops:
            luv = l[uv_layer]
            if luv.select:
                face_edge_verts.append(luv)
            else:
                is_face_fully_sel = False

        allEdgeVerts.extend(face_edge_verts)
        if is_face_fully_sel:
            if len(f.verts) != 4:
                nonQuadFaces.append(f)
                edgeVerts.extend(face_edge_verts)
            else:
                selFaces.append(f)
                # fill dictionary
                for l in f.loops:
                    luv = l[uv_layer]
                    x = round(luv.uv.x, precision)
                    y = round(luv.uv.y, precision)
                    vertsDict[(x, y)].append(luv)
        else:
            edgeVerts.extend(face_edge_verts)

    # If no “fully selected edges” at all, just fall back to allEdgeVerts
    noEdge = False
    if not edgeVerts:
        noEdge = True
        edgeVerts.extend(allEdgeVerts)

    # If no faces are fully selected, we create the filtered list
    if not selFaces:
        for ev in edgeVerts:
            # avoid duplicates if the same coordinate is repeated
            if not list_contains_quasi(filteredVerts, ev):
                filteredVerts.append(ev)
    else:
        filteredVerts = edgeVerts

    return edgeVerts, filteredVerts, selFaces, nonQuadFaces, vertsDict, noEdge

def fill_verts_dict_line(uv_layer, bm, selVerts, vertsDict):
    """
    For line-based selections, fill a dictionary of (x,y) -> [UV-loops].
    """
    for f in bm.faces:
        for l in f.loops:
            luv = l[uv_layer]
            if luv.select:
                x = round(luv.uv.x, precision)
                y = round(luv.uv.y, precision)
                vertsDict[(x, y)].append(luv)

def are_verts_aligned_on_x_or_y(verts):
    """
    Check if all the selected verts share the same X or the same Y (within a tiny threshold).
    """
    if not verts:
        return False
    allowed = 1e-5
    v0 = verts[0].uv
    same_x = True
    same_y = True
    for v in verts:
        if abs(v.uv.x - v0.x) > allowed:
            same_x = False
        if abs(v.uv.y - v0.y) > allowed:
            same_y = False
        if not (same_x or same_y):
            return False
    return True

def make_equal_distance_line(verts, vertsDict, start_uv=None):
    """
    Distribute a line of selected UV vertices so that they have equal spacing
    from first to last.
    """
    # Sort by X
    verts_sorted = sorted(verts, key=lambda x: x.uv.x)

    first = verts_sorted[0].uv
    last = verts_sorted[-1].uv

    # Determine if “more horizontal” or “more vertical”
    dx = (last.x - first.x)
    dy = (last.y - first.y)

    horizontal = True
    if abs(dx) > 1e-5:
        slope = dy / dx
        if abs(slope) > 1:
            horizontal = False
    else:
        horizontal = False

    # If horizontal, line them up by X; else line them up by Y
    if horizontal:
        length = hypot(dx, dy)
        if start_uv == last:
            # if user’s chosen “start” is the last
            currentX = last.x - length
            currentY = last.y
        else:
            currentX = first.x
            currentY = first.y

        number = len(verts_sorted)
        spacing = length / (number - 1) if number > 1 else length

        for v in verts_sorted:
            old_x = round(v.uv.x, precision)
            old_y = round(v.uv.y, precision)
            # move all dictionary entries for that coordinate:
            for vert_loop in vertsDict[(old_x, old_y)]:
                vert_loop.uv.x = currentX
                vert_loop.uv.y = currentY
            currentX += spacing
    else:
        # sort by Y descending
        verts_sorted = sorted(verts, key=lambda x: x.uv.y, reverse=True)
        first = verts_sorted[0].uv
        last = verts_sorted[-1].uv

        length = hypot(first.x - last.x, first.y - last.y)
        if start_uv == last:
            currentX = last.x
            currentY = last.y + length
        else:
            currentX = first.x
            currentY = first.y

        number = len(verts_sorted)
        spacing = length / (number - 1) if number > 1 else length

        for v in verts_sorted:
            old_x = round(v.uv.x, precision)
            old_y = round(v.uv.y, precision)
            for vert_loop in vertsDict[(old_x, old_y)]:
                vert_loop.uv.x = currentX
                vert_loop.uv.y = currentY
            currentY -= spacing

def scale_to_axis_zero(verts, vertsDict, startv=None, forced_horizontal=None):
    """
    If the selected UVs are not strictly along X or Y, we:
    - position the 2D cursor at `startv`
    - scale them to zero along X or Y axis
    """
    sorted_x = sorted(verts, key=lambda x: x.uv.x)
    first = sorted_x[0]
    last = sorted_x[-1]

    dx = last.uv.x - first.uv.x
    dy = last.uv.y - first.uv.y
    horizontal = forced_horizontal if forced_horizontal is not None else True
    if abs(dx) > 1e-5:
        slope = dy / dx
        if abs(slope) > 1:
            horizontal = False
    else:
        horizontal = False

    if startv is None:
        startv = first

    # set 2D cursor to that
    snap_cursor_to_closest_selected([startv])

    # scale to 0 on Y or X
    if horizontal:
        do_scale_zero('Y')
    else:
        do_scale_zero('X')

def do_scale_zero(axis='Y'):
    """
    Using the transform operator in the Image Editor to scale selection to zero on X or Y.
    """
    last_area_type = bpy.context.area.type
    bpy.context.area.type = 'IMAGE_EDITOR'
    last_pivot = bpy.context.space_data.pivot_point
    bpy.context.space_data.pivot_point = 'CURSOR'

    if axis.upper() == 'Y':
        bpy.ops.transform.resize(
            value=(1, 0, 1),
            constraint_axis=(False, True, False),
            mirror=False,
            proportional_edit_falloff='SMOOTH',
            proportional_size=1
        )
    else:
        bpy.ops.transform.resize(
            value=(0, 1, 1),
            constraint_axis=(True, False, False),
            mirror=False,
            proportional_edit_falloff='SMOOTH',
            proportional_size=1
        )

    bpy.context.space_data.pivot_point = last_pivot
    bpy.context.area.type = last_area_type

# -----------------------------------------------------------------------------
#  Rectangular Faces (Quads)
# -----------------------------------------------------------------------------

def shape_face(uv_layer, operator, face):
    """
    Attempt to shape the active face into a rectangle or square. We gather corners,
    reorder them, and then do MakeUvFaceEqualRectangle (the actual alignment).
    """
    corners = [l[uv_layer] for l in face.loops]
    if len(corners) != 4:
        return  # skip

    lucv, ldcv, rucv, rdcv = sort_face_corners(corners)
    startv = cursor_closest_to([lucv, ldcv, rdcv, rucv])
    # Default is by-shape, so “square” param is not forced inside here:
    make_uv_face_equal_rectangle(
        lucv, rucv, rdcv, ldcv, startv
    )

def sort_face_corners(corner_loops):
    """
    Sorts a 4-loop quad’s corners into (leftUp, leftDown, rightUp, rightDown).
    Returns them as loop objects that have .uv
    """
    # Convert to a normal python list
    corners = list(corner_loops)
    if len(corners) != 4:
        return corners[0], corners[0], corners[0], corners[0]

    # Find top 2, then figure out left vs right
    highest1 = max(corners, key=lambda c: c.uv.y)
    corners.remove(highest1)
    highest2 = max(corners, key=lambda c: c.uv.y)
    corners.remove(highest2)

    if highest1.uv.x < highest2.uv.x:
        leftUp, rightUp = highest1, highest2
    else:
        leftUp, rightUp = highest2, highest1

    # Now the other two are presumably lower
    low1 = corners[0]
    low2 = corners[1]
    if low1.uv.x < low2.uv.x:
        leftDown, rightDown = low1, low2
    else:
        leftDown, rightDown = low2, low1

    return leftUp, leftDown, rightUp, rightDown

def make_uv_face_equal_rectangle(lucv, rucv, rdcv, ldcv, startv, force_square=False):
    """
    Align the 4 corners of a quad to form a “nice rectangle”. If `force_square=True`,
    try to make them an even square in UV space.

    This version attempts to keep the original lengths along edges so that the rectangle
    can preserve shape. If you want strictly square, pass `force_square=True`.
    """
    # Acquire the length in X and Y from the chosen corner
    # so we treat that corner as “active”.
    # For now, we rely on standard hypot to measure.
    from math import hypot

    def distance(a, b):
        return hypot(a.x - b.x, a.y - b.y)

    # Figure out which corner is actually the “start corner” based on which
    # matches the ‘startv’ object:
    start_coord = startv.uv
    # Just store the .uv
    lu = lucv.uv
    ru = rucv.uv
    rd = rdcv.uv
    ld = ldcv.uv

    # identify which corner is the “closest” or “same” as startv
    # We'll just check an epsilon:
    all_corners = [lu, ru, rd, ld]
    # If it’s none of them, default to the first corner (lu).
    chosen = lu
    eps = 1e-5
    for c in all_corners:
        if abs(c.x - start_coord.x) < eps and abs(c.y - start_coord.y) < eps:
            chosen = c
            break

    # Now compute finalScaleX, finalScaleY, and top-left corner coords
    if chosen == lu:
        finalScaleX = distance(lu, ru)
        finalScaleY = distance(lu, ld)
        baseX = lu.x
        baseY = lu.y
        signX = 1.0
        signY = -1.0
    elif chosen == ru:
        finalScaleX = distance(ru, lu)
        finalScaleY = distance(ru, rd)
        baseX = ru.x - finalScaleX
        baseY = ru.y
        signX = 1.0
        signY = -1.0
    elif chosen == rd:
        finalScaleX = distance(rd, ld)
        finalScaleY = distance(rd, ru)
        baseX = rd.x - finalScaleX
        baseY = rd.y
        signX = 1.0
        signY = 1.0
    else:  # chosen == ld
        finalScaleX = distance(ld, rd)
        finalScaleY = distance(ld, lu)
        baseX = ld.x
        baseY = ld.y
        signX = 1.0
        signY = 1.0

    if force_square:
        finalScaleY = finalScaleX

    # Now set the .uv for each corner by dictionary
    # create a helper so we can update duplicates if any
    def set_uvs(loop, x, y):
        loop.uv.x = x
        loop.uv.y = y

    # top-left
    set_uvs(lucv, baseX, baseY)
    # top-right
    set_uvs(rucv, baseX + signX * finalScaleX, baseY)
    # bottom-right
    set_uvs(rdcv, baseX + signX * finalScaleX, baseY + signY * (-finalScaleY))
    # bottom-left
    set_uvs(ldcv, baseX, baseY + signY * (-finalScaleY))

# -----------------------------------------------------------------------------
#  “Follow Active Quads” logic
# -----------------------------------------------------------------------------

def follow_active_uv(operator, me, f_act, faces, EXTEND_MODE='LENGTH_AVERAGE'):
    """
    Adapted from ideasman42's uvcalc_follow_active.py with slight modifications.
    If `EXTEND_MODE='EVEN'`, we keep edges the same length, effectively “squaring” them.
    """
    bm = bmesh.from_edit_mesh(me)
    uv_act = bm.loops.layers.uv.active

    def walk_face_init(faces, f_act):
        # Tag everything True so it won't be walked on, then mark selected faces as False
        for ff in bm.faces:
            ff.tag = True
        for ff in faces:
            ff.tag = False
        f_act.tag = True  # starting face = done

    def walk_face(start_face):
        start_face.tag = True
        queue = [start_face]
        while queue:
            f0 = queue.pop()
            for l in f0.loops:
                e = l.edge
                # walk across manifold edges that are not seams
                if e.is_manifold and not e.seam:
                    l_other = l.link_loop_radial_next
                    f_next = l_other.face
                    if f_next.tag == False:
                        # yield triple: (current_face, loop, next_face)
                        yield (f0, l, f_next)
                        f_next.tag = True
                        queue.append(f_next)

    def walk_edge_loop(l):
        """
        Walk the edge loop starting from loop l until we come back or cannot proceed.
        """
        first_e = l.edge
        e = None
        while True:
            e = l.edge
            yield e

            if e.is_manifold:
                # move around the quad
                l = l.link_loop_radial_next
                if len(l.face.verts) == 4:
                    l = l.link_loop_next.link_loop_next
                    if l.edge == first_e:
                        break
                else:
                    break
            else:
                break

    # Possibly gather average lengths
    if EXTEND_MODE == 'LENGTH_AVERAGE':
        bm.edges.index_update()
        edge_lengths = [None] * len(bm.edges)

        for f in faces:
            if len(f.loops) != 4:
                continue
            loops = f.loops[:]
            pairA = (loops[0], loops[2])
            pairB = (loops[1], loops[3])

            for pair in (pairA, pairB):
                if edge_lengths[pair[0].edge.index] is None:
                    edge_length_store = [-1.0]
                    total_len = 0.0
                    count = 0
                    for ll in pair:
                        for e_loop in walk_edge_loop(ll):
                            if edge_lengths[e_loop.index] is None:
                                edge_lengths[e_loop.index] = edge_length_store
                                total_len += e_loop.calc_length()
                                count += 1
                    edge_length_store[0] = total_len / count if count else 1.0

    # Now do the main iteration
    walk_face_init(faces, f_act)

    def apply_uv(f_prev, l_prev, f_next):
        l_a = [None]*4
        l_b = [None]*4

        # loops in the “previous face”
        l_a[0] = l_prev
        l_a[1] = l_a[0].link_loop_next
        l_a[2] = l_a[1].link_loop_next
        l_a[3] = l_a[2].link_loop_next

        # loops in the “next face”
        l_next = l_prev.link_loop_radial_next
        # check winding
        if l_next.vert != l_prev.vert:
            l_b[1] = l_next
            l_b[0] = l_b[1].link_loop_next
            l_b[3] = l_b[0].link_loop_next
            l_b[2] = l_b[3].link_loop_next
        else:
            l_b[0] = l_next
            l_b[1] = l_b[0].link_loop_next
            l_b[2] = l_b[1].link_loop_next
            l_b[3] = l_b[2].link_loop_next

        la_uv = [la[uv_act].uv for la in l_a]
        lb_uv = [lb[uv_act].uv for lb in l_b]

        # compute scale factor
        if EXTEND_MODE == 'LENGTH_AVERAGE':
            try:
                fac = edge_lengths[l_b[2].edge.index][0] / edge_lengths[l_a[1].edge.index][0]
            except (ZeroDivisionError, TypeError):
                fac = 1.0
        elif EXTEND_MODE == 'EVEN':
            # a quick approach: force scale=1 so that it “copies” the shape
            fac = 1.0
        else:
            fac = 1.0

        def extrapolate_uv(factor, a_outer, a_inner, b_outer, b_inner):
            b_inner[:] = a_inner
            b_outer[:] = a_inner + ((a_inner - a_outer) * factor)

        # corners correspond: (3->0), (2->1)
        # top loops
        extrapolate_uv(fac, la_uv[3], la_uv[0], lb_uv[3], lb_uv[0])
        # bottom loops
        extrapolate_uv(fac, la_uv[2], la_uv[1], lb_uv[2], lb_uv[1])

    for tri in walk_face(f_act):
        apply_uv(*tri)

    bmesh.update_edit_mesh(me, loop_triangles=False)

# -----------------------------------------------------------------------------
#  Utility / Helpers
# -----------------------------------------------------------------------------

def list_contains_quasi(vert_list, candidate, eps=1e-5):
    """Check if candidate is 'almost' in vert_list, by XY coords."""
    for v in vert_list:
        if (abs(v.uv.x - candidate.uv.x) < eps and abs(v.uv.y - candidate.uv.y) < eps):
            return True
    return False

def snap_cursor_to_closest_selected(filteredVerts):
    """
    Snap the 2D cursor to the first vertex in the list (or do a more robust
    distance check if you want).
    """
    if not filteredVerts:
        return
    # We’ll just pick the first for simplicity; or you can pick the truly "closest to current cursor"
    v = filteredVerts[0]
    set_all_2d_cursors(v.uv.x, v.uv.y)

def cursor_closest_to(verts):
    """
    Return whichever vertex is closest to the current 2D cursor location,
    ignoring image dimension scaling. If no area found, fallback to first.
    """
    if not verts:
        return None
    # see if there's an actual Image Editor:
    # If none found, just return the first.
    min_dist = float('inf')
    closest = verts[0]
    screen_areas = getattr(bpy.context, 'screen', None)
    if not screen_areas:
        return closest

    # If we do find an image editor, read the cursor_location
    # and compare. Typically in 2.8+ it's in [0..1].
    for area in screen_areas.areas:
        if area.type == 'IMAGE_EDITOR':
            loc = area.spaces.active.cursor_location
            for v in verts:
                dx = loc.x - v.uv.x
                dy = loc.y - v.uv.y
                d = (dx*dx + dy*dy)**0.5
                if d < min_dist:
                    min_dist = d
                    closest = v
            return closest
    return closest

def set_all_2d_cursors(x, y):
    """
    Set the UV editor’s 2D cursor to (x,y). If multiple image-editor areas exist, set them all.
    """
    last_area_type = bpy.context.area.type
    bpy.context.area.type = 'IMAGE_EDITOR'

    bpy.ops.uv.cursor_set(location=(x, y))

    bpy.context.area.type = last_area_type

def finish_success(me, start_time):
    """
    Wrap-up function that updates the edit mesh and prints an optional log.
    """
    bmesh.update_edit_mesh(me, loop_triangles=False)
    elapsed = round(timer() - start_time, 3)
    if elapsed > 0.05:
        print(f"UvSquares finished in {elapsed} sec.")

def deselect_all():
    bpy.ops.uv.select_all(action='DESELECT')

# -----------------------------------------------------------------------------
#  Extra Operators for “Rip” / “Join”
# -----------------------------------------------------------------------------

def rip_uv_faces(context, operator):
    """
    “Rip” UV faces apart: if a face is fully selected, convert them
    so only one loop remains selected, etc.
    """
    start_time = timer()
    obj = context.active_object
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    uv_layer = bm.loops.layers.uv.verify()

    selected_faces = []
    for f in bm.faces:
        if all(l[uv_layer].select for l in f.loops):
            selected_faces.append(f)

    if not selected_faces:
        # If nothing is fully selected, pick the first selected UV loop only
        target_loop = None
        for f in bm.faces:
            for l in f.loops:
                luv = l[uv_layer]
                if luv.select:
                    target_loop = luv
                    break
            if target_loop:
                break

        # Deselect everything, then select that single
        deselect_all()
        if target_loop:
            target_loop.select = True

        finish_success(me, start_time)
        return

    # If we do have some fully selected faces, “rip” them
    deselect_all()
    for f in selected_faces:
        for l in f.loops:
            l[uv_layer].select = True

    finish_success(me, start_time)

def join_uv_faces(context, operator):
    """
    “Join” selection to the nearest unselected vertices. Must be very close.
    This uses a small radius to unify them.
    """
    start_time = timer()
    obj = context.active_object
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    uv_layer = bm.loops.layers.uv.verify()

    vertsDict = defaultdict(list)

    # radius for joining
    radius = 0.002

    # gather selected
    for f in bm.faces:
        for l in f.loops:
            luv = l[uv_layer]
            if luv.select:
                x = round(luv.uv.x, precision)
                y = round(luv.uv.y, precision)
                vertsDict[(x, y)].append(luv)

    # attempt to unify with “closest” unselected
    for key, group in vertsDict.items():
        min_dist = 9999.0
        min_loop = None
        start_uv = group[0].uv
        for f in bm.faces:
            for l in f.loops:
                luv = l[uv_layer]
                if not luv.select:
                    dx = start_uv.x - luv.uv.x
                    dy = start_uv.y - luv.uv.y
                    d = (dx*dx + dy*dy)**0.5
                    if d < min_dist and d < radius:
                        min_dist = d
                        min_loop = luv
        if min_loop:
            # unify all in group to that min_loop.uv
            for uv_loop in group:
                uv_loop.uv.x = min_loop.uv.x
                uv_loop.uv.y = min_loop.uv.y

    finish_success(me, start_time)

# -----------------------------------------------------------------------------
#  Operators & Panel
# -----------------------------------------------------------------------------

class UV_PT_UvSquares(bpy.types.Operator):
    """Reshapes UV faces to a grid of equivalent squares"""
    bl_idname = "uv.uv_squares"
    bl_label = "UVs to grid of squares"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'

    def execute(self, context):
        main(context, self, square=True)
        return {'FINISHED'}

class UV_PT_UvSquaresByShape(bpy.types.Operator):
    """Reshapes UV faces to a grid with respect to shape by length of edges around selected corner"""
    bl_idname = "uv.uv_squares_by_shape"
    bl_label = "UVs to grid (respect shape)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'

    def execute(self, context):
        main(context, self, square=False)
        return {'FINISHED'}

class UV_PT_RipFaces(bpy.types.Operator):
    """Rip UV faces apart"""
    bl_idname = "uv.uv_face_rip"
    bl_label = "UV face rip"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'

    def execute(self, context):
        rip_uv_faces(context, self)
        return {'FINISHED'}

class UV_PT_JoinFaces(bpy.types.Operator):
    """Join selection to closest nonselected vertices (must be very close)"""
    bl_idname = "uv.uv_face_join"
    bl_label = "UV selection join to closest unselected"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'

    def execute(self, context):
        join_uv_faces(context, self)
        return {'FINISHED'}

class UV_PT_SnapToAxis(bpy.types.Operator):
    """Snap sequenced vertices to Axis"""
    bl_idname = "uv.uv_snap_to_axis"
    bl_label = "UV snap vertices to axis"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'

    def execute(self, context):
        main(context, self)
        return {'FINISHED'}

class UV_PT_SnapToAxisWithEqual(bpy.types.Operator):
    """Snap sequenced vertices to Axis with Equal Distance between them"""
    bl_idname = "uv.uv_snap_to_axis_and_equal"
    bl_label = "UV snap w/ equal distance"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'

    def execute(self, context):
        # run it twice to ensure it scales & then sets spacing
        main(context, self)
        main(context, self)
        return {'FINISHED'}

# The panel in the Image Editor sidebar (“N” panel)
class UV_PT_UvSquaresPanel(bpy.types.Panel):
    """UvSquares Panel"""
    bl_label = "UV Squares"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'UV Squares'

    def draw(self, context):
        layout = self.layout

        row = layout.row()
        row.label(text="Select Sequenced Vertices to:")
        split = layout.split()
        col = split.column(align=True)
        col.operator(UV_PT_SnapToAxis.bl_idname, text="Snap to Axis (X or Y)", icon="ARROW_LEFTRIGHT")
        col.operator(UV_PT_SnapToAxisWithEqual.bl_idname, text="Snap w/ Equal Distance", icon="THREE_DOTS")

        row = layout.row()
        row.label(text="Convert 4-corner faces to:")
        split = layout.split()
        col = split.column(align=True)
        col.operator(UV_PT_UvSquaresByShape.bl_idname, text="Grid by Shape", icon="UV_FACESEL")
        col.operator(UV_PT_UvSquares.bl_idname, text="Square Grid", icon="GRID")

        row = layout.row()
        row.label(text="Additional Tools:")
        col = layout.column(align=True)
        col.operator(UV_PT_JoinFaces.bl_idname, text="Join to Closest Unselected", icon="SNAP_GRID")
        col.operator(UV_PT_RipFaces.bl_idname, text="Rip UV Faces", icon="UV_SYNC_SELECT")

# Add menu entries (optional)
def menu_func_uv_squares(self, context):
    self.layout.operator(UV_PT_UvSquares.bl_idname, text="To Square Grid")

def menu_func_uv_squares_by_shape(self, context):
    self.layout.operator(UV_PT_UvSquaresByShape.bl_idname, text="Grid by Shape")

def menu_func_face_join(self, context):
    self.layout.operator(UV_PT_JoinFaces.bl_idname, text="Join to Closest Unselected")

# Store keymaps so we can remove them later
addon_keymaps = []

def register():
    bpy.utils.register_class(UV_PT_UvSquaresPanel)
    bpy.utils.register_class(UV_PT_UvSquares)
    bpy.utils.register_class(UV_PT_UvSquaresByShape)
    bpy.utils.register_class(UV_PT_RipFaces)
    bpy.utils.register_class(UV_PT_JoinFaces)
    bpy.utils.register_class(UV_PT_SnapToAxis)
    bpy.utils.register_class(UV_PT_SnapToAxisWithEqual)

    # Append to IMAGE_EDITOR > UV menu
    bpy.types.IMAGE_MT_uvs.append(menu_func_uv_squares)
    bpy.types.IMAGE_MT_uvs.append(menu_func_uv_squares_by_shape)
    bpy.types.IMAGE_MT_uvs.append(menu_func_face_join)

    # Example of adding a keymap in the UV Editor (ALT+E in UV Editor)
    wm = bpy.context.window_manager
    if wm.keyconfigs.addon:
        km = wm.keyconfigs.addon.keymaps.new(name='UV Editor', space_type='IMAGE_EDITOR')
        kmi = km.keymap_items.new(UV_PT_UvSquaresByShape.bl_idname, 'E', 'PRESS', alt=True)
        addon_keymaps.append((km, kmi))

def unregister():
    # Remove classes
    bpy.utils.unregister_class(UV_PT_SnapToAxisWithEqual)
    bpy.utils.unregister_class(UV_PT_SnapToAxis)
    bpy.utils.unregister_class(UV_PT_JoinFaces)
    bpy.utils.unregister_class(UV_PT_RipFaces)
    bpy.utils.unregister_class(UV_PT_UvSquaresByShape)
    bpy.utils.unregister_class(UV_PT_UvSquares)
    bpy.utils.unregister_class(UV_PT_UvSquaresPanel)

    # Remove menu items
    bpy.types.IMAGE_MT_uvs.remove(menu_func_uv_squares)
    bpy.types.IMAGE_MT_uvs.remove(menu_func_uv_squares_by_shape)
    bpy.types.IMAGE_MT_uvs.remove(menu_func_face_join)

    # Remove keymaps
    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()

if __name__ == "__main__":
    register()
