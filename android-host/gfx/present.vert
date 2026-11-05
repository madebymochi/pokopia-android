#version 450
// Present composite (nvn_vk.c): one textured quad per draw (triangle strip of 4), placed in the window's logical (landscape) NDC rectangle dst,
// then turned to the panel's native orientation (misc.x: the swapchain pre-transform, 1 = 90, 2 = 180, 3 = 270 degrees).
layout(push_constant) uniform PC { vec4 dst; vec4 src; vec4 misc; } pc;
layout(location = 0) out vec2 uv;
void main() {
    vec2 c = vec2(gl_VertexIndex & 1, gl_VertexIndex >> 1);
    vec2 p = mix(pc.dst.xy, pc.dst.zw, c);
    uv = mix(pc.src.xy, pc.src.zw, c);
    int r = int(pc.misc.x);
    if (r == 1) p = vec2(-p.y, p.x); else if (r == 2) p = -p; else if (r == 3) p = vec2(p.y, -p.x);
    gl_Position = vec4(p, 0, 1);
}
