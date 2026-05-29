"""
Main Orchestrator for 4-Case Regression Analysis

This script runs all experiments and generates comprehensive results
for the supply chain demand forecasting thesis.

Usage:
    python main.py                    # Run all cases
    python main.py --cases 1 2        # Run specific cases
    python main.py --no-neural        # Skip neural networks
    python main.py --tune             # Enable hyperparameter tuning
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import Config
from config.constants import RESULTS_DIR, FIGURES_DIR
from experiments import Case1Experiment, Case2Experiment, Case3Experiment, Case4Experiment
from evaluation.metrics import MetricsCalculator
from evaluation.visualizations import Visualizer


def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger('MainOrchestrator')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run 4-Case Regression Analysis for Supply Chain Forecasting'
    )
    
    parser.add_argument(
        '--cases', 
        type=int, 
        nargs='+', 
        default=[1, 2, 3, 4],
        choices=[1, 2, 3, 4],
        help='Cases to run (default: all)'
    )
    
    parser.add_argument(
        '--no-classical',
        action='store_true',
        help='Skip classical statistical models'
    )
    
    parser.add_argument(
        '--no-ml',
        action='store_true',
        help='Skip machine learning models'
    )
    
    parser.add_argument(
        '--no-neural',
        action='store_true',
        help='Skip neural network models (FFNN)'
    )
    
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Enable hyperparameter tuning (slower)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to disk'
    )
    
    return parser.parse_args()


def run_experiment(case_num: int, config: Config, logger: logging.Logger, save_results: bool) -> pd.DataFrame:
    """
    Run a single case experiment.
    
    Args:
        case_num: Case number (1-4)
        config: Configuration object
        logger: Logger instance
        
    Returns:
        DataFrame with results
    """
    experiment_classes = {
        1: Case1Experiment,
        2: Case2Experiment,
        3: Case3Experiment,
        4: Case4Experiment
    }
    
    if case_num not in experiment_classes:
        logger.error(f"Invalid case number: {case_num}")
        return None
    
    logger.info(f"\n{'='*60}")
    logger.info(f"STARTING CASE {case_num} EXPERIMENT")
    logger.info(f"{'='*60}\n")
    
    try:
        experiment_class = experiment_classes[case_num]
        experiment = experiment_class(config)
        results = experiment.run(save_results=save_results)
        
        logger.info(f"\nCase {case_num} Results:")
        logger.info(f"\n{results.to_string()}")
        
        return results
        
    except Exception as e:
        logger.error(f"Case {case_num} failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def generate_summary_report(all_results: dict, config: Config, logger: logging.Logger):
    """
    Generate a comprehensive summary report across all cases.
    
    Args:
        all_results: Dictionary of {case_name: results_df}
        config: Configuration object
        logger: Logger instance
    """
    logger.info("\n" + "="*60)
    logger.info("GENERATING SUMMARY REPORT")
    logger.info("="*60 + "\n")
    
    # Combine all results
    combined_results = []
    for case_name, results in all_results.items():
        if results is not None and not results.empty:
            results = results.copy()
            results['case'] = case_name
            combined_results.append(results)
    
    if not combined_results:
        logger.warning("No results to summarize")
        return
    
    full_results = pd.concat(combined_results, ignore_index=True)
    
    # Save combined results
    results_path = config.results_dir / 'all_cases_results.csv'
    full_results.to_csv(results_path, index=False)
    logger.info(f"Combined results saved to: {results_path}")
    
    # Generate summary statistics
    summary = full_results.groupby('case').agg({
        'MAE': ['min', 'mean'],
        'RMSE': ['min', 'mean'],
        'R2': ['max', 'mean']
    }).round(4)
    
    logger.info("\nSummary by Case:")
    logger.info(f"\n{summary.to_string()}")
    
    # Find best models per case
    logger.info("\nBest Models per Case (by RMSE):")
    for case in full_results['case'].unique():
        case_results = full_results[full_results['case'] == case]
        best_idx = case_results['RMSE'].idxmin()
        best_model = case_results.loc[best_idx, 'model']
        best_rmse = case_results.loc[best_idx, 'RMSE']
        logger.info(f"  {case}: {best_model} (RMSE: {best_rmse:.4f})")
    
    # Generate LaTeX table for thesis
    latex_path = config.results_dir / 'results_table.tex'
    latex_table = full_results.pivot_table(
        values=['MAE', 'RMSE', 'R2'],
        index='model',
        columns='case',
        aggfunc='mean'
    ).round(4)
    
    with open(latex_path, 'w') as f:
        f.write("% Auto-generated results table\n")
        f.write(latex_table.to_latex())
    logger.info(f"\nLaTeX table saved to: {latex_path}")
    
    # Save summary as JSON
    summary_dict = {
        'timestamp': datetime.now().isoformat(),
        'cases_run': list(all_results.keys()),
        'total_models': len(full_results),
        'best_per_case': {}
    }
    
    for case in full_results['case'].unique():
        case_results = full_results[full_results['case'] == case]
        best_idx = case_results['RMSE'].idxmin()
        summary_dict['best_per_case'][case] = {
            'model': case_results.loc[best_idx, 'model'],
            'MAE': float(case_results.loc[best_idx, 'MAE']),
            'RMSE': float(case_results.loc[best_idx, 'RMSE']),
            'R2': float(case_results.loc[best_idx, 'R2'])
        }
    
    json_path = config.results_dir / 'summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary_dict, f, indent=2)
    logger.info(f"Summary JSON saved to: {json_path}")


def generate_cross_case_figures(all_results: dict, config: Config, logger: logging.Logger):
    """
    Generate cross-case comparison figures.
    
    Args:
        all_results: Dictionary of results
        config: Configuration object
        logger: Logger instance
    """
    logger.info("\nGenerating cross-case comparison figures...")
    
    visualizer = Visualizer(output_dir=config.figures_dir)
    
    # Combine results
    combined_results = []
    for case_name, results in all_results.items():
        if results is not None and not results.empty:
            results = results.copy()
            results['case'] = case_name
            combined_results.append(results)
    
    if not combined_results:
        return
    
    full_results = pd.concat(combined_results, ignore_index=True)
    
    # Figure 7.10: Model Ranking Heatmap
    try:
        pivot = full_results.pivot_table(
            values='RMSE',
            index='model',
            columns='case',
            aggfunc='mean'
        )
        
        visualizer.plot_metrics_heatmap(
            metrics_df=pivot,
            title='Model Performance Comparison (RMSE)',
            filename='figure_7_10_model_ranking_heatmap',
            chapter='chapter7',
            metric_name='RMSE'
        )
        logger.info("Generated model ranking heatmap")
    except Exception as e:
        logger.error(f"Heatmap generation failed: {e}")

    # Figure 7.10b: Model Performance Heatmap (R2)
    try:
        pivot_r2 = full_results.pivot_table(
            values='R2',
            index='model',
            columns='case',
            aggfunc='mean'
        )

        visualizer.plot_metrics_heatmap(
            metrics_df=pivot_r2,
            title='Model Performance Comparison (R2)',
            filename='figure_7_10_model_r2_heatmap',
            chapter='chapter7',
            metric_name='R2'
        )
        logger.info("Generated R2 heatmap")
    except Exception as e:
        logger.error(f"R2 heatmap generation failed: {e}")


def main():
    """Main entry point for the orchestrator."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    logger.info("="*60)
    logger.info("4-CASE REGRESSION ANALYSIS FOR SUPPLY CHAIN FORECASTING")
    logger.info("="*60)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Cases to run: {args.cases}")
    
    # Create configuration
    config = Config()
    config.run_classical_models = not args.no_classical
    config.run_ml_models = not args.no_ml
    config.run_neural_networks = not args.no_neural
    config.cases_to_run = args.cases
    
    if args.no_classical:
        logger.info("Classical models: DISABLED")
    if args.no_ml:
        logger.info("ML models: DISABLED")
    if args.no_neural:
        logger.info("Neural networks: DISABLED")
    
    # Run experiments
    all_results = {}
    
    for case_num in args.cases:
        results = run_experiment(case_num, config, logger, save_results=not args.no_save)
        all_results[f'case{case_num}'] = results
    
    # Generate summary report
    if not args.no_save:
        generate_summary_report(all_results, config, logger)
        generate_cross_case_figures(all_results, config, logger)
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT COMPLETED")
    logger.info("="*60)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    successful = sum(1 for r in all_results.values() if r is not None and not r.empty)
    logger.info(f"Successfully completed: {successful}/{len(args.cases)} cases")
    
    if not args.no_save:
        logger.info(f"\nResults saved to: {config.results_dir}")
        logger.info(f"Figures saved to: {config.figures_dir}")
    
    return all_results


if __name__ == "__main__":
    main()
