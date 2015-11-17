// Bits in here inspired by https://github.com/nukep/rust-opengl-util/blob/master/shader.rs
extern crate sdl2;
extern crate nalgebra;
use sdl2::keyboard::Keycode;
use sdl2::event::Event;
use gl::types::GLuint;
use gl::types::GLsizei;
use gl::types::GLint;
use gl::types::GLfloat;
use std::ffi::CString;
use std::mem::size_of;

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
                if log_len == 0 {
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
// XXX this is ugly imo. Make it more like Program
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
        if !successful {
            let mut log_len = 0;
            unsafe { glctx.GetShaderiv(s.id, gl::INFO_LOG_LENGTH, &mut log_len) };
            if log_len <= 0 {
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

fn p_matrix(l:GLfloat, r:GLfloat, t:GLfloat, b:GLfloat, n:GLfloat, f:GLfloat) -> [GLfloat; 16] {
    [
        (2. * n)/(r - l), 0., (r + l)/(r - l), 0.,
        0., (2. * n)/(t - b), (t + b)/(t - b), 0.,
        0., 0., -(f + n) / (f - n), (-2. * f * n)/(f - n),
        0., 0., -1., 0.
    ]
}

fn main() {
    println!("OK let's do this!");
    let sctx = sdl2::init().unwrap();
    let vctx = sctx.video().unwrap();
    let wctx = vctx.window("Some Bullshit", 800, 600)
        .opengl()
        .build()
        .unwrap();
    let sdl_glctx = wctx.gl_create_context().unwrap();
    let mut running = true;
    let mut event_pump = sctx.event_pump().unwrap();
    let glctx = &gl::Gl::load_with(|s| vctx.gl_get_proc_address(s));
    let vertices = [
        //Front face
        -1.0, -1.0,  1.0,
         1.0, -1.0,  1.0,
         1.0,  1.0,  1.0, ];/*
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
    ];*/

    let indices = [
        0, 1, 2,];      /*0, 2, 3,    // Front face
        4, 5, 6,      4, 6, 7,    // Back face
        8, 9, 2,      8, 2, 11,   // Top face
        12, 13, 14,   12, 14, 15, // Bottom face
        16, 17, 18,   16, 18, 19, // Right face
        20, 21, 22,   20, 22, 23  // Left face
    ];*/
    let colors = [
        //Front face
         1.0,  0.0,  0.0, 1.0,
         0.0,  1.0,  0.0, 1.0,
         0.0,  0.0,  1.0, 1.0, ]; /*
         1.0,  1.0,  1.0, 1.0,
    ];
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
    */
    let pm //p_matrix(10., -10., 10., -10., -5., 5.);
     = [
        1., 0., 0., 0.,
        0., 1., 0., 0.,
        0., 0., 1., 0.,
        0., 0., 0., 1.,
    ]; 
    let mvm = [0.5, 0., 0., 0.,
               0., 0.5, 0., 0., 
               0., 0., 0.5, 0.,
               0., 0., 0.5, 1.];
    let pr = unsafe { // Initialize opengl
        glctx.ClearColor(0.0, 0.0, 0.0, 1.0);
        glctx.ClearDepth(1.0);
        glctx.Enable(gl::DEPTH_TEST);
        glctx.DepthFunc(gl::LEQUAL);
        //glctx.ShadeModel(gl::SMOOTH);
        //glctx.Hint(gl::PERSPECTIVE_CORRECTION_HINT, gl::NICEST);
        let vs = Shader::new(glctx, gl::VERTEX_SHADER,
        r#"
            attribute vec4 aColor;
            attribute vec3 aPosition;
            uniform uBlock {
                mat4 uMVMatrix;
                mat4 uPMatrix;
            };

            varying vec4 vColor;

            void main(void) {
                gl_Position = uPMatrix * uMVMatrix * vec4(aPosition, 1.0);
                vColor = aColor;
            }
        "#).unwrap();
        let fs = Shader::new(glctx, gl::FRAGMENT_SHADER,
        r#"
            varying vec4 vColor;
            void main(void) {
                gl_FragColor = vColor;
            }
        "#).unwrap();
        Program::new(glctx, &[fs, vs]).unwrap()
    };
    // Get attr and uniform locations
    let vloc = unsafe {glctx.GetProgramResourceLocation(pr.id, gl::PROGRAM_INPUT, CString::new("aPosition").unwrap().as_ptr())};
    let cloc = unsafe { glctx.GetProgramResourceLocation(pr.id, gl::PROGRAM_INPUT, CString::new("aColor").unwrap().as_ptr())};
    let ubloc = unsafe { glctx.GetProgramResourceIndex(pr.id, gl::UNIFORM_BLOCK, CString::new("uBlock").unwrap().as_ptr())} as u32;
    println!("Color loc: {}, Vertex loc: {}, ub loc: {}", cloc, vloc, ubloc);
    if cloc == -1 {
        panic!("couldn't find color");
    }
    if vloc == -1 {
        panic!("couldn't find position");
    }
    if ubloc == -1 {
        panic!("couldn't find uniform block");
    }
    let bufs = unsafe {
        let mut bs: [u32;4] = [0, 0, 0, 0];
        glctx.CreateBuffers(4, bs.as_mut_ptr());
        bs
    };
    let vbuf = bufs[0];
    let cbuf = bufs[1];
    let ibuf = bufs[2];
    let ubbuf = bufs[3];
    if (vbuf == 0 || cbuf == 0 || ibuf == 0 || ubbuf == 0) {
        panic!("Could not CreateBuffers, got v:{} c:{} i:{} ub:{}", vbuf, cbuf, ibuf, ubbuf);
    }
    println!("CreateBuffers'd, got v:{} c:{} i:{}", vbuf, cbuf, ibuf);
    let mflags = gl::MAP_WRITE_BIT | gl::MAP_PERSISTENT_BIT | gl::MAP_COHERENT_BIT;
    let sflags = mflags;// | gl::DYNAMIC_STORAGE_BIT;
    let ubptr = unsafe {
        let mut data_size = 0 as GLint;
        let params = gl::BUFFER_DATA_SIZE;
        glctx.GetProgramResourceiv(pr.id, gl::UNIFORM_BLOCK, ubloc, 1, &gl::BUFFER_DATA_SIZE, 1, std::ptr::null_mut(), &mut data_size as *mut GLint);
        println!("UB size: {}", data_size);
        glctx.NamedBufferStorage(ubbuf, data_size as i64, std::ptr::null(), sflags);
        let ptr = glctx.MapNamedBufferRange(ubbuf, 0, data_size as i64, mflags) as *mut f32;
        if (ptr as u64 == 0) {
            panic!("Failed to map Uniform buffer {}", glctx.GetError());
        }
        std::ptr::write(ptr as *mut [f32; 16], mvm);
        std::ptr::write(ptr.offset(16) as *mut [f32; 16], pm);
        println!("Ptr: {}, offset: {}", ptr as u64, ptr.offset(16) as u64);
        ptr
    };
    let vptr = unsafe {
        // Map and fill the vertex buffer
        glctx.NamedBufferStorage(vbuf, vertices.len() as i64 * 4, std::ptr::null(), sflags);
        let ptr = glctx.MapNamedBufferRange(vbuf, 0, vertices.len() as i64 * 4, mflags);
        if (ptr as u64 == 0) {
            panic!("Failed to map vertex buffer {}", glctx.GetError());
        }
        std::ptr::write(ptr as *mut [f32; 9], vertices);
        ptr
    };
    println!("Mapped vertex buffer to {}", vptr as u64);
    let cptr = unsafe {
        // Map and fill the color buffer
        glctx.NamedBufferStorage(cbuf, colors.len() as i64 * 4, std::ptr::null(), sflags);
        let ptr = glctx.MapNamedBufferRange(cbuf, 0, colors.len() as i64 * 4, mflags);
        if (ptr as u64 == 0) {
            panic!("Failed to map color buffer {}", glctx.GetError());
        }
        std::ptr::write(ptr as *mut [f32; 12], colors);
        ptr
    };
    let iptr = unsafe {
        // Map and fill the index buffer
        glctx.NamedBufferStorage(ibuf, indices.len() as i64 * 4, std::ptr::null(), sflags);
        let ptr = glctx.MapNamedBufferRange(ibuf, 0, indices.len() as i64 * 4, mflags);
        if (ptr as u64 == 0) {
            panic!("Failed to map color buffer {}", glctx.GetError());
        }
        std::ptr::write(ptr as *mut [u32; 3], indices);
        ptr
    };
    println!("Mapped color buffer to {}", cptr as u64);

    unsafe {
        //glctx.ProgramUniformMatrix4fv(pr.id, pmloc, 1, false as u8, pm.as_ptr());
        //glctx.ProgramUniformMatrix4fv(pr.id, mvmloc, 1, false as u8, mvm.as_ptr());
        glctx.UniformBlockBinding(pr.id, ubloc as u32, 1);
    };

    let vao = unsafe {
        // Create a Vertex Array Object 
        let mut v: u32 = 0;
        glctx.CreateVertexArrays(1, (&mut v));
        glctx.VertexArrayAttribFormat(v, vloc as u32, 3, gl::FLOAT, 0, 0);
        glctx.VertexArrayAttribFormat(v, cloc as u32, 4, gl::FLOAT, 0, 0);
        v
    };
    unsafe {
        // Tie the shit together
        glctx.EnableVertexArrayAttrib(vao, vloc as u32);
        glctx.EnableVertexArrayAttrib(vao, cloc as u32);
        glctx.VertexArrayVertexBuffer(vao, vloc as u32, vbuf, 0, 4);
        glctx.VertexArrayVertexBuffer(vao, cloc as u32, cbuf, 0, 4);
        glctx.VertexArrayElementBuffer(vao, ibuf);
    };
    if (vao == 0) {
        panic!("Failed to CreateVertexArrays");
    }
    println!("Created VAO: {}", vao);
    // Dump the viewport cause whynot
    let vp = unsafe {
        let mut bs: [GLint;4] = [0, 0, 0, 0];
        glctx.GetIntegerv(gl::VIEWPORT, bs.as_mut_ptr());
        bs
    };
    let vp_x = vp[0];
    let vp_y = vp[1];
    let vp_w = vp[2];
    let vp_h = vp[3];
    println!("Viewport at ({}, {}), sized {}x{}", vp_x, vp_y, vp_w, vp_h);

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
            glctx.UseProgram(pr.id);
            glctx.BindVertexArray(vao);
            glctx.BindBufferBase(gl::UNIFORM_BUFFER, 1, ubbuf);
            glctx.MemoryBarrier(gl::CLIENT_MAPPED_BUFFER_BARRIER_BIT);
            glctx.DrawElements(gl::TRIANGLES, indices.len() as i32, gl::UNSIGNED_INT, std::ptr::null());
            glctx.Finish();
        }
        wctx.gl_swap_window();
    }
}
