import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker
# Set the backend to Qt5Agg for interactive plotting
matplotlib.use('Agg')

# Apply Modern Computer (Computer Modern) font to all text elements
plt.rcParams.update({
    'mathtext.fontset': 'cm',  # Use Computer Modern
    'font.family': 'serif',    # Apply serif font family to all text
    'mathtext.rm': 'serif',    # Use serif for regular math text
    'mathtext.it': 'serif:italic',  # Use serif italic for italic math text
    'mathtext.bf': 'serif:bold',  # Use serif bold for bold math text
    'font.size': 12,  # Set general font size
})
# Define a function to format the y-tick labels
def format_ytick(y, _):
    # Adjust tolerance for rounding
    tolerance = 1e-5
    if y == 0:
        return r'$0$'  # Specifically handle the case where y is exactly 0
    elif abs(y) < 1e-3 or abs(y) > 1e3:
        return f'${{\\mathrm{{{y:.2e}}}}}$'  # Scientific notation for very small or large numbers
    elif abs(y - round(y)) < tolerance:
        return f'${{\\mathrm{{{int(y)}}}}}$'  # Integer formatting
    elif abs(y - round(y, 1)) < tolerance:
        return f'${{\\mathrm{{{y:.1f}}}}}$'  # One decimal place if close to a single decimal
    elif abs(y - round(y, 2)) < tolerance:
        return f'${{\\mathrm{{{y:.2f}}}}}$'  # Two decimal places if close to two decimals
    elif abs(y - round(y, 3)) < tolerance:
        return f'${{\\mathrm{{{y:.3f}}}}}$'  # Three decimal places if close to three decimals
    else:
        return f'${{\\mathrm{{{y:.4f}}}}}$'  # Default to four decimal places

def gp_plotfitness(gp):
    if gp.config['runcontrol']['plot']['fitness']:
        """
        Plot the fitness vs. generations and update the plot with each generation.
        """
        num_generations = gp.config['runcontrol']['num_generations']
        num_pop = gp.config['runcontrol']['num_pop']

        # Determine the number of subplots based on the available datasets
        num_plots = 1
        if gp.userdata['xval'] is not None:
            num_plots += 1
        if gp.userdata['xtest'] is not None:
            num_plots += 1

        # Pre-calculate axis limits based on initial values
        min_fitness = min(np.min(gp.state['mean_fitness']['ensemble']['train'] - gp.state['std_fitness']['ensemble']['train']),
                          np.min([np.min(gp.state['mean_fitness']['isolated']['train'][id_pop] - gp.state['std_fitness']['isolated']['train'][id_pop]) for id_pop in range(num_pop)]))
        max_fitness = max(np.max(gp.state['mean_fitness']['ensemble']['train'] + gp.state['std_fitness']['ensemble']['train']),
                          np.max([np.max(gp.state['mean_fitness']['isolated']['train'][id_pop] + gp.state['std_fitness']['isolated']['train'][id_pop]) for id_pop in range(num_pop)]))

        if gp.userdata['xval'] is not None:
            min_fitness = min(min_fitness, np.min(gp.state['mean_fitness']['ensemble']['validation'] - gp.state['std_fitness']['ensemble']['validation']),
                              np.min([np.min(gp.state['mean_fitness']['isolated']['validation'][id_pop] - gp.state['std_fitness']['isolated']['validation'][id_pop]) for id_pop in range(num_pop)]))
            max_fitness = max(max_fitness, np.max(gp.state['mean_fitness']['ensemble']['validation'] + gp.state['std_fitness']['ensemble']['validation']),
                              np.max([np.max(gp.state['mean_fitness']['isolated']['validation'][id_pop] + gp.state['std_fitness']['isolated']['validation'][id_pop]) for id_pop in range(num_pop)]))

        if gp.userdata['xtest'] is not None:
            min_fitness = min(min_fitness, np.min(gp.state['mean_fitness']['ensemble']['test'] - gp.state['std_fitness']['ensemble']['test']),
                              np.min([np.min(gp.state['mean_fitness']['isolated']['test'][id_pop] - gp.state['std_fitness']['isolated']['test'][id_pop]) for id_pop in range(num_pop)]))
            max_fitness = max(max_fitness, np.max(gp.state['mean_fitness']['ensemble']['test'] + gp.state['std_fitness']['ensemble']['test']),
                              np.max([np.max(gp.state['mean_fitness']['isolated']['test'][id_pop] + gp.state['std_fitness']['isolated']['test'][id_pop]) for id_pop in range(num_pop)]))

        min_fitness = 0.95 * min_fitness
        max_fitness = 1.05 * max_fitness

        # Initialize the fitness vs generation plot if not already done
        if 'fitness_generation' not in gp.plot:
            gp.plot['fitness_generation'] = {}
            gp.plot['fitness_generation']['fig'], gp.plot['fitness_generation']['axes'] = plt.subplots(num_plots, 1, figsize=(8, 4 * num_plots))

        fig = gp.plot['fitness_generation']['fig']
        axes = gp.plot['fitness_generation']['axes']

        if num_plots == 1:
            axes = [axes]  # Ensure axes is a list even if there's only one plot

        # Titles for the subplots
        titles = ['Train']
        if gp.userdata['xval'] is not None:
            titles.append('Validation')
        if gp.userdata['xtest'] is not None:
            titles.append('Test')

        # Data keys for accessing different fitness types
        data_keys = ['train']
        if gp.userdata['xval'] is not None:
            data_keys.append('validation')
        if gp.userdata['xtest'] is not None:
            data_keys.append('test')
        
        sns.set(style="darkgrid")
        colors = sns.color_palette(['#ff6347', '#4682b4', '#32cd32'])  # Tomato, SteelBlue, LimeGreen
        # colors = ['#e41a1c', '#377eb8', '#4daf4a']  # Manually specified distinct colors
        # colors = sns.color_palette("husl", num_pop + 1)
        # colors = ['#e41a1c', '#377eb8', '#4daf4a']  # Red, Blue, Green
        # colors = sns.color_palette("muted", num_pop + 1)  # Use 'muted' 'bright' for distinct colors
        

        # Plot for each subplot (train, validation, test)
        for i, (ax, title, key) in enumerate(zip(axes, titles, data_keys)):
            # Set lighter gray background for the plotting area
            ax.set_facecolor('#D3D3D3')  # Lighter gray background
            
            # Set the spines (box) color to black
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(0.5)  # Adjust the width as needed
        
            ax.clear()
            generations = np.arange(gp.state['generation'] + 1)
            
            
            ax.grid(True, which='both', color='white', linestyle='-', linewidth=0.3)
            
            if num_pop > 1:
                # Plot ensemble fitness
                ax.plot(generations, gp.state['best']['fitness']['ensemble'][key], label=r"$\mathrm{Best\ Ensemble}$", linewidth=2, color=colors[0])
                ax.fill_between(generations,
                                gp.state['mean_fitness']['ensemble'][key] - gp.state['std_fitness']['ensemble'][key],
                                gp.state['mean_fitness']['ensemble'][key] + gp.state['std_fitness']['ensemble'][key],
                                color=colors[0], alpha=0.3, label=r"$\mathrm{Mean}\pm\mathrm{Std\ Ensemble}$")
    
                # Plot isolated populations' fitness
                for id_pop in range(num_pop):
                    ax.plot(generations, gp.state['best']['fitness']['isolated'][key][id_pop],
                            label=r"$\mathrm{Best\ Pop\ %d}$" % (id_pop + 1), linewidth=2, color=colors[id_pop + 1])
                    ax.fill_between(generations,
                                    gp.state['mean_fitness']['isolated'][key][id_pop] - gp.state['std_fitness']['isolated'][key][id_pop],
                                    gp.state['mean_fitness']['isolated'][key][id_pop] + gp.state['std_fitness']['isolated'][key][id_pop],
                                    color=colors[id_pop + 1], alpha=0.3, label=r"$\mathrm{Mean}\pm\mathrm{Std\ Pop\ %d}$" % (id_pop + 1))
            else:
                # Plot fitness
                ax.plot(generations, gp.state['best']['fitness']['ensemble'][key], label=r"$\mathrm{Best\ Individual}$", linewidth=2, color=colors[0])
                ax.fill_between(generations,
                                gp.state['mean_fitness']['ensemble'][key] - gp.state['std_fitness']['ensemble'][key],
                                gp.state['mean_fitness']['ensemble'][key] + gp.state['std_fitness']['ensemble'][key],
                                color=colors[0], alpha=0.3, label=r"$\mathrm{Mean}\pm\mathrm{Std\ Population}$")

            # Apply axis limits and labels
            ax.set_xlim((0, num_generations))
            ax.set_ylim((min_fitness, max_fitness))
            ax.set_ylabel(r"$\mathrm{Fitness}$", fontsize=12)
            ax.legend(loc="best", fontsize=8)

            # Get the position of the y-axis label
            ylabel_position = ax.yaxis.label.get_position()

            # Calculate the position adjustment needed for centering
            title_length = len(title)
            title_offset = (title_length / 2.0) * 0.03  # Adjust 0.02 based on title length and rotation

            # Set the title at the adjusted position
            ax.set_title(rf"$\mathrm{{{title}}}$", fontsize=12, loc='right', rotation=270, x=1.05, y=ylabel_position[1] - title_offset)

            # Set the x-axis ticks to be integers and adjust tick label font size
            max_ticks = 10  # Max number of ticks to display on x-axis
            tick_interval = max(1, num_generations // max_ticks)
            ax.set_xticks(np.arange(0, num_generations + 1, step=tick_interval))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(format_ytick))
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # Set tick labels using LaTeX formatting
            ax.set_xticklabels([f"${{\\mathrm{{{x}}}}}$" for x in ax.get_xticks()], fontdict={'fontsize': 10})
            # ax.set_yticklabels([f"${{\\mathrm{{{y}}}}}$" for y in ax.get_yticks()], fontdict={'fontsize': 10})
            
            # Show the x-axis label on every subplot
            ax.set_xlabel(r"$\mathrm{Generation}$", fontsize=12)
            ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)

        # Adjust layout to prevent overlap of axis labels
        plt.tight_layout()

        # Update the plot and process the event loop
        # plt.show(block=False)
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Save the figure in the specified formats
        for fmt in gp.config['runcontrol']['plot']['format']:
            fig.savefig(gp.config['runcontrol']['plot']['folder'] + f"{gp.runname}_FitnessVsGeneration.{fmt}", dpi=300, format=fmt)

        # plt.pause(0.1)
        plt.close(fig)
