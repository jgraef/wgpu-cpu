mod tests;
pub mod util;

use clap::{
    Parser,
    Subcommand,
};
use color_eyre::eyre::Error;
use dotenvy::dotenv;
use owo_colors::OwoColorize;

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
    },
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

            let mut test_results = vec![];

            tests.for_names(&names, |test| {
                println!("{} {}", "Running".green(), test.name());

                let test_result = test.run()?;
                if let Some(message) = test_result.error_message() {
                    println!("Test {}", "failed".red());
                    println!("{message}");
                    println!("");
                }
                test_results.push((test, test_result));

                Ok(())
            })?;

            println!("");
            println!("{}", "Summary".bold());
            for (test, test_result) in test_results {
                if let Some(message) = test_result.error_message() {
                    let error_line = message.lines().next().unwrap_or("<empty error message>");
                    println!(
                        "  Test {} {}: {}",
                        "failed".red(),
                        test.name(),
                        error_line.trim()
                    );
                }
                else {
                    println!("  Test {}: {}", "passed".green(), test.name());
                }
            }
        }
        Command::GenerateReference {
            tests: names,
            force,
        } => {
            if names.is_empty() && !force {
                println!(
                    "Are you sure to regenerate all reference images? Use `--force` if you're sure."
                );
            }
            else {
                tests.for_names(&names, |test| {
                    tracing::info!(test = test.name(), "Generating reference");
                    test.generate_reference(Default::default())?;
                    Ok(())
                })?;
            }
        }
    }

    Ok(())
}
