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
    eyre,
};
use image::{
    ImageReader,
    RgbaImage,
};

use crate::util::check_eq_image;

mod colored_triangle;

#[derive(Debug)]
pub struct TestFunction {
    pub name: &'static str,
    pub renderer: fn() -> RgbaImage,
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

    pub fn run(&self) -> Result<(), Error> {
        let reference = self.files.load_reference(self.test.name)?;

        let rendered = catch_unwind(|| (self.test.renderer)()).map_err(|error| {
            if let Some(error) = error.downcast_ref::<String>() {
                eyre!("{error}")
            }
            else if let Some(error) = error.downcast_ref::<&'static str>() {
                eyre!("{error}")
            }
            else {
                resume_unwind(error);
            }
        })?;

        self.files.save_output(self.test.name, &rendered)?;

        check_eq_image(&reference, &rendered)?;

        Ok(())
    }

    pub fn generate_reference(&self) -> Result<(), Error> {
        let reference = (self.test.renderer)();
        self.files.save_reference(self.test.name, &reference)?;
        Ok(())
    }
}

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

        let tests = inventory::iter::<TestFunction>()
            .map(|test| (test.name, test))
            .collect();

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
