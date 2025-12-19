mod tests;
pub mod util;

use clap::{
    Parser,
    Subcommand,
    ValueEnum,
};
use color_eyre::eyre::Error;
use dotenvy::dotenv;
use owo_colors::OwoColorize;
use wgpu_cpu::{
    Config,
    ShaderBackend,
};

use crate::tests::{
    TestFiles,
    Tests,
};

#[derive(Debug, Parser)]
struct Args {
    #[clap(flatten)]
    files: TestFiles,

    #[clap(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Test {
        tests: Vec<String>,
        #[clap(short, long)]
        no_unit_tests: bool,
    },
    GenerateReference {
        tests: Vec<String>,

        #[clap(short, long)]
        force: bool,

        #[clap(short = 'b')]
        shader_backend: Option<ShaderBackendArg>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Default)]
enum ShaderBackendArg {
    #[default]
    Interpreter,
    Compiler,
}

fn main() -> Result<(), Error> {
    let _ = dotenv();
    color_eyre::install()?;
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    let tests = Tests::new(args.files)?;

    match args.command {
        Command::Test {
            tests: names,
            no_unit_tests,
        } => {
            if !no_unit_tests {
                let mut cargo = std::process::Command::new("cargo")
                    .args(["test", "--all-features"])
                    .spawn()?;
                cargo.wait()?;
            }

            let mut results = vec![];

            tests.for_names(&names, |test| {
                println!("{} {}", "Running".green(), test.name());

                for (variant, result) in test.run()? {
                    if let Some(message) = result.error_message() {
                        println!("Test {} ({variant})", "failed".red());
                        println!("{message}");
                        println!("");
                    }
                    results.push((test, variant, result));
                }

                Ok(())
            })?;

            println!("");
            println!("{}", "Summary".bold());
            for (test, variant, result) in results {
                if let Some(message) = result.error_message() {
                    let error_line = message.lines().next().unwrap_or("<empty error message>");
                    println!(
                        "  Test {} {}/{variant}: {}",
                        "failed".red(),
                        test.name(),
                        error_line.trim()
                    );
                }
                else {
                    println!("  Test {}: {}/{variant}", "passed".green(), test.name());
                }
            }
        }
        Command::GenerateReference {
            tests: names,
            force,
            shader_backend,
        } => {
            if names.is_empty() && !force {
                println!(
                    "Are you sure to regenerate all reference images? Use `--force` if you're sure."
                );
            }
            else {
                let config = Config {
                    shader_backend: match shader_backend.unwrap_or_default() {
                        ShaderBackendArg::Interpreter => ShaderBackend::Interpreter,
                        ShaderBackendArg::Compiler => ShaderBackend::Compiler,
                    },
                };

                tests.for_names(&names, |test| {
                    tracing::info!(test = test.name(), "Generating reference");
                    test.generate_reference(config.clone())?;
                    Ok(())
                })?;
            }
        }
    }

    Ok(())
}
