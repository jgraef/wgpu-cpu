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
        Command::Test { tests: names } => {
            let mut results = vec![];

            tests.for_names(&names, |test| {
                println!("{} {}", "Running".green(), test.name());

                let result = test.run();

                match &result {
                    Ok(()) => {}
                    Err(error) => {
                        println!("Test {}", "failed".red());
                        println!("{error}");
                        println!("");
                    }
                }

                results.push((test, result));

                Ok(())
            })?;

            println!("");
            for (test, result) in results {
                match &result {
                    Ok(()) => {
                        println!("Test {} {}", "passed".green(), test.name());
                    }
                    Err(error) => {
                        let error = error.to_string();
                        let error_line = error.lines().next().unwrap_or_default();
                        println!(
                            "Test {} {}: {}",
                            "failed".red(),
                            test.name(),
                            error_line.trim()
                        );
                    }
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
                    test.generate_reference()?;
                    Ok(())
                })?;
            }
        }
    }

    Ok(())
}
