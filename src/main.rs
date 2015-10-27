// Bits in here inspired by https://github.com/nukep/rust-opengl-util/blob/master/shader.rs
extern crate sdl2;
use sdl2::keyboard::Keycode;
use sdl2::event::Event;
use sdl2::video::GLContext;
use gl::types::GLuint;
use gl::types::GLint;

mod gl {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub struct Shader<'a> {
    pub id: GLuint,
    pub glctx: &'a gl::Gl,
}

pub struct Program<'a> {
    pub id: GLuint,
    pub glctx: &'a gl::Gl,
}

impl<'a> Drop for Program<'a> {
    fn drop(&mut self) {
        unsafe {
            self.glctx.DeleteProgram(self.id);
        }
    }
}

impl<'a> Program<'a> {
    fn new(glctx: &'a gl::Gl, shaders: &[Shader]) -> Result<Program<'a>, String> {
        let p = Program {
            id: unsafe { glctx.CreateProgram() },
            glctx: glctx,
        };
        let successful: bool = unsafe {
            for s in shaders {
                glctx.AttachShader(p.id, s.id);
            }
            let mut result: GLint = 0;
            glctx.LinkProgram(p.id);
            glctx.GetProgramiv(p.id, gl::LINK_STATUS, &mut result);
            result == gl::TRUE as GLint
        };
        match successful {
            true => Ok(p),
            false => Err({
                let mut log_len = 0;
                unsafe {
                    glctx.GetProgramiv(p.id, gl::INFO_LOG_LENGTH, &mut log_len);
                }
                if (log_len == 0) {
                    String::from("No program link log :|")
                } else {
                    let mut buf = Vec::with_capacity(log_len as usize);
                    let buf_ptr = buf.as_mut_ptr() as *mut gl::types::GLchar;
                    unsafe {
                        glctx.GetProgramInfoLog(p.id, log_len, std::ptr::null_mut(), buf_ptr);
                        buf.set_len(log_len as usize);
                    }
                    match String::from_utf8(buf) {
                        Ok(log) => format!("LINKFAIL: {}", log),
                        Err(vec) => format!("Could not decode shader log {}", vec)
                    }
                }
            })
        }
    }
}
// IDFK what this is for. Looks like a destructor.
impl<'a> Drop for Shader<'a> {
    fn drop(&mut self) {
        unsafe {
            self.glctx.DeleteShader(self.id);
        };
    }
}
// XXX this is ugly as shit imo. Make it more like Program
impl<'a> Shader<'a> {
    fn new (glctx: &'a gl::Gl, typ: GLuint, source: &str) -> Result<Shader<'a>, String> {
        let s = Shader {
            id: unsafe { glctx.CreateShader(typ) },
            glctx: glctx,
        };
        let successful: bool = unsafe {
            let ptr: *const u8 = source.as_bytes().as_ptr();
            let ptr_i8: *const i8 = std::mem::transmute(ptr);
            let len = source.len() as GLint;
            glctx.ShaderSource(s.id, 1, &ptr_i8, &len);    
            glctx.CompileShader(s.id);
            let mut result: GLint = 0;
            glctx.GetShaderiv(s.id, gl::COMPILE_STATUS, &mut result);
            result == gl::TRUE as GLint
        };
        if (!successful) {
            let mut log_len = 0;
            unsafe { glctx.GetShaderiv(s.id, gl::INFO_LOG_LENGTH, &mut log_len) };
            if (log_len <= 0) {
                Err(String::from("No shader info log?"))
            } else {

                let mut buf = Vec::with_capacity(log_len as usize);
                let buf_ptr = buf.as_mut_ptr() as *mut gl::types::GLchar;
                unsafe {
                    glctx.GetShaderInfoLog(s.id, log_len, std::ptr::null_mut(), buf_ptr);
                    buf.set_len(log_len as usize);
                };

                match String::from_utf8(buf) {
                    Ok(log) => Err(format!("COMPILEFAIL: {}", log)),
                    Err(vec) => Err(format!("Could not convert compilation log from buffer: {}",vec)),
                }
            }
        } else {
            Ok(s)
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
    let glctx = &gl::Gl::load_with(|s| vctx.gl_get_proc_address(s));
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
        let vs = Shader::new(glctx, gl::VERTEX_SHADER,
        r#"
            attribute vec3 aVertexPosition;
            attribute vec4 aColor;

            uniform mat4 uMVMatrix;
            uniform mat4 uPMatrix;

            varying vec4 vColor;

            void main(void) {
                gl_Position = uPMatrix * uMVMatrix * vec4(aVertexPosition, 1.0);
                vColor = aColor;
            }
        "#).unwrap();
        let mut fs = Shader::new(glctx, gl::FRAGMENT_SHADER,
        r#"
            varying vec4 vColor;
            void main(void) {
                gl_FragColor = vColor;
            }
        "#).unwrap();
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
                _ => {println!("Unknown event")},
            }
        }
        unsafe {
            glctx.Clear(gl::COLOR_BUFFER_BIT|gl::DEPTH_BUFFER_BIT|gl::STENCIL_BUFFER_BIT);
        }
        wctx.gl_swap_window();
    }
}
