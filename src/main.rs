// Bits in here inspired by (pasted from) https://github.com/nukep/rust-opengl-util/blob/master/shader.rs
extern crate sdl2;
use sdl2::keyboard::Keycode;
use sdl2::event::Event;
use sdl2::video::GLContext;
use gl::types::GLuint;
use gl::types::GLint;

mod gl {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
pub struct Shader {
    pub id: GLuint,
    pub glctx: gl::Gl,
}
// IDFK what this is for. Looks like a destructor.
impl Drop for Shader {
    fn drop(&mut self) {
        unsafe {self.glctx.DeleteShader(self.id)};
    }
}
impl Shader {
    fn new (ctx: gl::Gl, typ: GLuint) -> Shader {
        Shader {
            glctx: ctx,
            id: unsafe { ctx.CreateShader(typ) }
        }
    }
    pub fn source(&mut self, source: &str) {
        unsafe {
            let ptr: *const u8 = source.as_bytes().as_ptr();
            let ptr_i8: *const i8 = std::mem::transmute(ptr);
            let len = source.len() as GLint;
            self.glctx.ShaderSource(self.id, 1, &ptr_i8, &len);    
        };
        let successful = unsafe {
            self.glctx.CompileShader(self.id);
            let mut result: GLint = 0;
            self.glctx.GetShaderiv(self.id, gl::COMPILE_STATUS, &mut result);
            result != 0
        };
        if (!successful) {
            let mut len = 0;
            unsafe { self.glctx.GetShaderiv(self.id, gl::INFO_LOG_LENGTH, &mut len) };
            assert!(len > 0);

            let mut buf = Vec::with_capacity(len as usize);
            let buf_ptr = buf.as_mut_ptr() as *mut gl::types::GLchar;
            unsafe {
                self.glctx.GetShaderInfoLog(self.id, len, std::ptr::null_mut(), buf_ptr);
                buf.set_len(len as usize);
            };

            match String::from_utf8(buf) {
                Ok(log) => println!("COMPILEFAIL: {}", log),
                Err(vec) => panic!("Could not convert compilation log from buffer: {}",vec)
            }
        }
    }
}
fn main() {
    println!("OK let's do this!");
    let sctx = sdl2::init().unwrap();
    let vctx = sctx.video().unwrap();
    let wctx = vctx.window("Some Bullshit", 800, 600)
        .opengl()
        .build()
        .unwrap();
    let glctx = wctx.gl_create_context().unwrap();
    let mut running = true;
    let mut event_pump = sctx.event_pump().unwrap();
    let glctx = gl::Gl::load_with(|s| vctx.gl_get_proc_address(s));
    let vertices:Vec<f32> = vec![
        //Front face
        -1.0, -1.0,  1.0,
         1.0, -1.0,  1.0,
         1.0,  1.0,  1.0,
        -1.0,  1.0,  1.0,

        //Back face
        -1.0, -1.0, -1.0,
        -1.0,  1.0, -1.0,
         1.0,  1.0, -1.0,
         1.0, -1.0, -1.0,

        //Top face
        -1.0,  1.0, -1.0,
        -1.0,  1.0,  1.0,
         1.0,  1.0,  1.0,
         1.0,  1.0, -1.0,

        //Bottom face
        -1.0, -1.0, -1.0,
         1.0, -1.0, -1.0,
         1.0, -1.0,  1.0,
        -1.0, -1.0,  1.0,

        //Right face
         1.0, -1.0, -1.0,
         1.0,  1.0, -1.0,
         1.0,  1.0,  1.0,
         1.0, -1.0,  1.0,

        //Left face
        -1.0, -1.0, -1.0,
        -1.0, -1.0,  1.0,
        -1.0,  1.0,  1.0,
        -1.0,  1.0, -1.0
    ];

    let indices: Vec<u16> = vec![
        0, 1, 2,      0, 2, 3,    // Front face
        4, 5, 6,      4, 6, 7,    // Back face
        8, 9, 2,      8, 2, 11,   // Top face
        12, 13, 14,   12, 14, 15, // Bottom face
        16, 17, 18,   16, 18, 19, // Right face
        20, 21, 22,   20, 22, 23  // Left face
    ];
    let colors:Vec<f32> = vec![
        //Front face
         1.0,  1.0,  1.0, 1.0,
         1.0,  1.0,  1.0, 1.0,
         1.0,  1.0,  1.0, 1.0,
         1.0,  1.0,  1.0, 1.0,

        //Back face
         1.0,  1.0,  1.0, 1.0,
         1.0,  1.0,  1.0, 1.0,
         1.0,  1.0,  1.0, 1.0,
         1.0,  1.0,  1.0, 1.0,

        //Top face
         1.0,  1.0,  1.0, 1.0,
         1.0,  1.0,  1.0, 1.0,
         1.0,  1.0,  1.0, 1.0,
         1.0,  1.0,  1.0, 1.0,

        //Bottom face
         1.0,  1.0,  1.0, 1.0,
         1.0,  1.0,  1.0, 1.0,
         1.0,  1.0,  1.0, 1.0,
         1.0,  1.0,  1.0, 1.0,

        //Right face
         1.0,  1.0,  1.0, 1.0,
         1.0,  1.0,  1.0, 1.0,
         1.0,  1.0,  1.0, 1.0,
         1.0,  1.0,  1.0, 1.0,

        //Left face
         1.0,  1.0,  1.0, 1.0,
         1.0,  1.0,  1.0, 1.0,
         1.0,  1.0,  1.0, 1.0,
         1.0,  1.0,  1.0, 1.0,
    ];
    unsafe { // Initialize opengl
        glctx.ClearColor(1.0, 0.0, 0.0, 1.0);
        glctx.ClearDepth(1.0);
        glctx.Enable(gl::DEPTH_TEST);
        glctx.DepthFunc(gl::LEQUAL);
        //glctx.ShadeModel(gl::SMOOTH);
        //glctx.Hint(gl::PERSPECTIVE_CORRECTION_HINT, gl::NICEST);
        let vs = Shader::new(glctx, gl::VERTEX_SHADER);
        vs.source(r#"
            attribute vec3 aVertexPosition;
            attribute vec4 aColor;

            uniform mat4 uMVMatrix;
            uniform mat4 uPMatrix;

            varying vec4 vColor;

            void main(void) {
                gl_Position = uPMatrix * uMVMatrix * vec4(aVertexPosition, 1.0);
                vColor = aColor;
        }"#);
        //glctx.ShaderSource(vs, 1, 
        let fs = Shader::new(glctx, gl::FRAGMENT_SHADER);
        fs.source(r#"
            varying vec4 vColor;
            void main(void) {
                gl_FragColor = vColor;
            }
        "#);
        let pr = glctx.CreateProgram();
        glctx.AttachShader(pr, vs.id);
        glctx.AttachShader(pr, fs.id);
        glctx.LinkProgram(pr);
        glctx.UseProgram(pr);
    }

    while running {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit {..} | Event::KeyDown { keycode: Some(Keycode::Q), .. } => 
                    {running = false},
                Event::KeyDown { keycode: Some(kc), timestamp, .. } => {println!("Got a {} at {}", kc, timestamp)},
                _ => {println!("Unknown event")}
            }
        }
        unsafe {
            glctx.Clear(gl::COLOR_BUFFER_BIT|gl::DEPTH_BUFFER_BIT|gl::STENCIL_BUFFER_BIT);
        }
        wctx.gl_swap_window();
    }
}
