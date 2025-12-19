use std::{
    collections::HashMap,
    panic::{
        catch_unwind,
        resume_unwind,
    },
    path::PathBuf,
};

use color_eyre::eyre::{
    Error,
    bail,
    eyre,
};
use image::{
    ImageReader,
    RgbaImage,
};
use wgpu_cpu::{
    Config,
    ShaderBackend,
};

use crate::util::{
    check_eq_image,
    create_device_and_queue,
};

mod colored_triangle;

const ALL_SHADER_BACKENDS: &[ShaderBackend] = &[
    wgpu_cpu::ShaderBackend::Interpreter,
    wgpu_cpu::ShaderBackend::Compiler,
];

#[derive(Debug)]
pub struct TestFunction {
    pub name: &'static str,
    pub renderer: fn(wgpu::Device, wgpu::Queue) -> RgbaImage,
    pub shader_backends: &'static [ShaderBackend],
}

impl TestFunction {
    pub fn shader_backends(&self) -> impl Iterator<Item = ShaderBackend> {
        if self.shader_backends.is_empty() {
            ALL_SHADER_BACKENDS
        }
        else {
            self.shader_backends
        }
        .into_iter()
        .copied()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Test<'files> {
    test: &'static TestFunction,
    files: &'files TestFiles,
}

impl<'files> Test<'files> {
    pub fn name(&self) -> &'static str {
        self.test.name
    }

    pub fn run(&self, test_config: &TestConfig) -> Result<TestResults, Error> {
        let reference = self.files.load_reference(self.test.name)?;
        let mut test_results = vec![];

        let shader_backends = self
            .test
            .shader_backends()
            .filter(|backend| test_config.shader_backends.contains(backend));

        for shader_backend in shader_backends {
            let config = Config { shader_backend };
            let test_result = self.run_with(&reference, config)?;
            test_results.push((format!("{shader_backend:?}"), test_result));
        }

        Ok(test_results)
    }

    pub fn run_with(&self, reference: &RgbaImage, config: Config) -> Result<TestResult, Error> {
        let render_result = catch_unwind(|| {
            let (device, queue) = create_device_and_queue(config);
            (self.test.renderer)(device, queue)
        });

        let test_result = match render_result {
            Ok(rendered) => {
                self.files.save_output(self.test.name, &rendered)?;
                if let Err(error) = check_eq_image(reference, &rendered) {
                    TestResult::ImageTestFailed(error)
                }
                else {
                    TestResult::Passed
                }
            }
            Err(panic) => {
                let message = if let Some(message) = panic.downcast_ref::<String>() {
                    message.to_string()
                }
                else if let Some(message) = panic.downcast_ref::<&'static str>() {
                    message.to_string()
                }
                else {
                    resume_unwind(panic);
                };
                TestResult::RenderPanic(message)
            }
        };

        Ok(test_result)
    }

    pub fn generate_reference(&self, config: Config) -> Result<(), Error> {
        let (device, queue) = create_device_and_queue(config);
        let reference = (self.test.renderer)(device, queue);
        self.files.save_reference(self.test.name, &reference)?;
        Ok(())
    }
}

#[derive(Debug)]
pub enum TestResult {
    Passed,
    ImageTestFailed(Error),
    RenderPanic(String),
}

#[allow(dead_code)]
impl TestResult {
    pub fn passed(&self) -> bool {
        matches!(self, Self::Passed)
    }

    pub fn failed(&self) -> bool {
        !self.passed()
    }

    pub fn error_message(&self) -> Option<String> {
        match self {
            TestResult::Passed => None,
            TestResult::ImageTestFailed(report) => Some(report.to_string()),
            TestResult::RenderPanic(panic) => Some(panic.clone()),
        }
    }
}

pub type TestResults = Vec<(String, TestResult)>;

#[derive(Debug, clap::Args)]
pub struct TestFiles {
    /// Path to reference files.
    #[clap(long, default_value = "tests/reference")]
    pub reference: PathBuf,

    /// Path to test output files.
    #[clap(long, default_value = "tests/output")]
    pub output: PathBuf,
}

impl TestFiles {
    fn load_reference(&self, test_name: &str) -> Result<RgbaImage, Error> {
        let file_name = format!("{}.png", test_name);
        let path = self.reference.join(&file_name);
        let image = ImageReader::open(&path)
            .map_err(|error| eyre!("Could not open reference file: {}: {error}", path.display()))?
            .decode()
            .map_err(|error| {
                eyre!(
                    "Could not decode reference image: {}: {error}",
                    path.display()
                )
            })?
            .to_rgba8();
        Ok(image)
    }

    fn save_reference(&self, test_name: &str, output: &RgbaImage) -> Result<(), Error> {
        let file_name = format!("{}.png", test_name);
        let path = self.reference.join(&file_name);
        output
            .save(&path)
            .map_err(|error| eyre!("Could not save reference: {}: {error}", path.display()))?;
        Ok(())
    }

    fn save_output(&self, test_name: &str, output: &RgbaImage) -> Result<(), Error> {
        let file_name = format!("{}.png", test_name);
        let path = self.output.join(&file_name);
        output
            .save(&path)
            .map_err(|error| eyre!("Could not save output: {}: {error}", path.display()))?;
        Ok(())
    }
}

inventory::collect!(TestFunction);

#[macro_export]
macro_rules! test {
    ($func:ident) => {
        inventory::submit! {crate::tests::TestFunction {
            name: stringify!($func),
            renderer: $func,
            shader_backends: &[],
        }}
    };
}

#[derive(Debug)]
pub struct Tests {
    tests: HashMap<&'static str, &'static TestFunction>,
    files: TestFiles,
}

impl Tests {
    pub fn new(files: TestFiles) -> Result<Self, Error> {
        std::fs::create_dir_all(&files.reference)?;
        std::fs::create_dir_all(&files.output)?;

        let mut tests = HashMap::new();
        for test in inventory::iter::<TestFunction>() {
            if tests.insert(test.name, test).is_some() {
                bail!("Multiple tests with the same name: {}", test.name);
            }
        }

        Ok(Self { tests, files })
    }

    pub fn for_names<'a, S>(
        &'a self,
        names: impl IntoIterator<Item = S>,
        mut f: impl FnMut(Test<'a>) -> Result<(), Error>,
    ) -> Result<(), Error>
    where
        S: AsRef<str>,
    {
        let mut at_least_one = false;
        for name in names {
            at_least_one = true;
            let name = name.as_ref();
            let test = self
                .tests
                .get(name)
                .ok_or_else(|| eyre!("Test not found: {name}"))?;
            f(Test {
                test: *test,
                files: &self.files,
            })?;
        }

        if !at_least_one {
            for test in self.tests.values() {
                f(Test {
                    test: *test,
                    files: &self.files,
                })?;
            }
        }

        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct TestConfig {
    pub shader_backends: Vec<ShaderBackend>,
}
