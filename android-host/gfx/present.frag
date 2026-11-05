#version 450
// misc.y 1: the game frame, opaque; 0: the controls layer, premultiplied alpha scaled by misc.z
layout(push_constant) uniform PC { vec4 dst; vec4 src; vec4 misc; } pc;
layout(set = 0, binding = 0) uniform sampler2D tex;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 o;
void main() {
    vec4 c = texture(tex, uv);
    o = pc.misc.y > 0.5 ? vec4(c.rgb, 1) : c * pc.misc.z;
}
