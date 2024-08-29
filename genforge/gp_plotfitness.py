import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend to Qt5Agg for interactive plotting

import matplotlib.pyplot as plt
import numpy as np

def gp_plotfitness(gp):
    if gp.config['runcontrol']['plotfitness']:
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

        colors = plt.cm.viridis(np.linspace(0, 1, num_pop))

        # Plot for each subplot (train, validation, test)
        for i, (ax, title, key) in enumerate(zip(axes, titles, data_keys)):
            ax.clear()
            generations = np.arange(gp.state['generation'] + 1)

            # Plot ensemble fitness
            ax.plot(generations, gp.state['best']['fitness']['ensemble'][key], label=r"$\mathrm{Best\ Ensemble}$", linewidth=2, color='blue')
            ax.fill_between(generations,
                            gp.state['mean_fitness']['ensemble'][key] - gp.state['std_fitness']['ensemble'][key],
                            gp.state['mean_fitness']['ensemble'][key] + gp.state['std_fitness']['ensemble'][key],
                            color='blue', alpha=0.3, label=r"$\mathrm{Mean}\pm\mathrm{Std\ Ensemble}$")

            # Plot isolated populations' fitness
            for id_pop in range(num_pop):
                ax.plot(generations, gp.state['best']['fitness']['isolated'][key][id_pop],
                        label=r"$\mathrm{Best\ Pop\ %d}$" % (id_pop + 1), linewidth=1, color=colors[id_pop])
                ax.fill_between(generations,
                                gp.state['mean_fitness']['isolated'][key][id_pop] - gp.state['std_fitness']['isolated'][key][id_pop],
                                gp.state['mean_fitness']['isolated'][key][id_pop] + gp.state['std_fitness']['isolated'][key][id_pop],
                                color=colors[id_pop], alpha=0.3, label=r"$\mathrm{Mean}\pm\mathrm{Std\ Pop\ %d}$" % (id_pop + 1))

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
            ax.set_xticks(np.arange(0, num_generations + 1, step=1))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
            ax.tick_params(axis='both', which='major', labelsize=8)

            # Show the x-axis label on every subplot
            ax.set_xlabel(r"$\mathrm{Generation}$", fontsize=12)
            ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)

        # Apply Modern Computer (Computer Modern) font to all text elements
        plt.rcParams.update({
            'mathtext.fontset': 'cm',  # Use Computer Modern
            'font.family': 'serif',    # Apply serif font family to all text
            'mathtext.rm': 'serif',    # Use serif for regular math text
            'mathtext.it': 'serif:italic', # Use serif italic for italic math text
            'mathtext.bf': 'serif:bold', # Use serif bold for bold math text
            'font.size': 12             # Set general font size
        })

        # Update the plot and process the event loop
        plt.show(block=False)
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Save the figure in the specified formats
        for fmt in gp.config['runcontrol']['plotformat']:
            fig.savefig(f"{gp.runname}_FitnessVsGeneration.{fmt}", dpi=300, format=fmt)

        plt.pause(0.1)
