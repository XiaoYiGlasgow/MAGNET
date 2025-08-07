@hydra.main(version_base=None, config_path="../../config/simulation", config_name="default_simulation")
def main(cfg: DictConfig) -> None:
    """
    Run GNN_Trans simulation for nuboard visualization.
    """

    # Import required modules
    from nuplan.planning.script.run_simulation import main as run_simulation
    from nuplan.planning.script.utils import set_default_path

    # Set default paths
    set_default_path()

    # Override planner configuration to use GNN_Trans
    cfg.planner = {
        '_target_': 'GNN_Trans.Nuboard.gnn_trans_planner.GNNTransPlanner',
        'model_path': '/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/GNN_Trans_params/best_model.pth'
    }

    # Set simulation parameters
    cfg.scenario_builder.scenario_filter.num_scenarios_per_type = 3
    cfg.scenario_builder.scenario_filter.limit_total_scenarios = 10

    # Set output directory
    output_dir = Path('/mnt/local/Xiao_Yi/gnn_trans_simulation_results')
    output_dir.mkdir(exist_ok=True)
    cfg.output_dir = str(output_dir)

    # Run simulation
    print("Starting GNN_Trans simulation for nuboard...")
    run_simulation(cfg)
    print(f"Simulation completed. Results saved to: {output_dir}")

