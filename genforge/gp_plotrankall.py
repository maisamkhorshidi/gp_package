import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

def compute_offset(x, base_value=0.0375):
    sign = -1 if x % 2 == 1 else 1
    multiplier = (x // 2) + 1
    return sign * multiplier * base_value

def gp_plotrankall(gp):
    if gp.config['runcontrol']['plot']['rankall']:
        num_generations = gp.config['runcontrol']['num_generations']
        num_pop = gp.config['runcontrol']['num_pop']
        pop_size = gp.config['runcontrol']['pop_size']
        gen = gp.state['generation']
        idx_en = gp.track['all_ensemble']['idx']

        num_plots = 1
        if gp.userdata['xval'] is not None:
            num_plots += 1
        if gp.userdata['xtest'] is not None:
            num_plots += 1

        # Initialize the fitness vs generation plot if not already done
        if 'rank_generation' not in gp.plot:
            gp.plot['rankall_generation'] = {}
            gp.plot['rankall_generation']['fig'], gp.plot['rankall_generation']['axes'] = plt.subplots(num_plots, 1, figsize=(8, 4 * num_plots))

        fig = gp.plot['rankall_generation']['fig']
        axes = gp.plot['rankall_generation']['axes']

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
        colors = sns.color_palette("bright", num_pop * pop_size)

        # Compute offset
        offset = np.zeros((num_pop))
        if num_pop < 10:
            for id_pop in range(num_pop):
                offset[id_pop] = compute_offset(id_pop)
        else:
            for id_pop in range(num_pop):
                offset[id_pop] = compute_offset(id_pop, base_value=0.75 / num_pop)

        # Plot for each subplot (train, validation, test)
        for ii, (ax, title, key) in enumerate(zip(axes, titles, data_keys)):
            # Set lighter gray background for the plotting area
            ax.set_facecolor('#D3D3D3')  # Lighter gray background
            
            # Set the spines (box) color to black
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(0.5)  # Adjust the width as needed
                
            # Compute rankings
            rank_iso_pop = gp.track['rank']['fitness']['isolated'][key]
            fit_en = gp.track['all_ensemble']['fitness'][key]
            rank_ensemble = [None for _ in range(num_pop)]
            for id_pop in range(num_pop):
                rank_ensemble[id_pop] = np.ones((pop_size, gen + 2), dtype=int) * (pop_size + 1)
                rank_ensemble[id_pop][:, 0] = np.array(list(range(1, pop_size + 1)), dtype=int)
                for id_gen in range(1, gen + 2):
                    if gp.config['runcontrol']['minimisation']:
                        sort_idx = list(np.argsort(fit_en[id_gen - 1]))
                    else:
                        sort_idx = list(np.argsort(-fit_en[id_gen - 1]))
                    idx_en_sort = idx_en[id_gen - 1][sort_idx, :]
                    for id_rank in range(pop_size):
                        id_iso = int(np.where(rank_iso_pop[id_gen - 1][id_pop] == id_rank)[0])
                        rank_en = int(np.min(np.where(idx_en_sort[:, id_pop] == id_iso)[0]))
                        if rank_en <= pop_size:
                            rank_ensemble[id_pop][id_rank, id_gen] = rank_en + 1

            ax.clear()
            generations = np.arange(-1, gp.state['generation'] + 1)

            counter = 0
            for id_pop in range(num_pop):
                for id_ind in range(pop_size):
                    # Plot ensemble fitness
                    ax.plot(generations, rank_ensemble[id_pop][id_ind, :] + offset[id_pop],
                            # label=fr"$\mathrm{{ER(IR_{{{id_pop+1}}}={id_ind+1})}}$",  # Commenting out legend labels
                            linewidth=1, color=colors[counter])
                    counter += 1

            # Apply axis limits and labels
            ax.set_xlim((-1, num_generations + 1))
            ax.set_ylim((pop_size + 1, 0))  # Reverse y-axis
            ax.set_ylabel(r"$\mathrm{Rank}$", fontsize=12)
            # ax.legend(loc="best", fontsize=8)  # Commenting out the legend

            # Grid at 1x1
            ax.set_xticks(np.arange(0, num_generations + 1, step=1))
            y_ticks = np.arange(1, pop_size + 2, step=1)
            ax.set_yticks(y_ticks[:-1])  # Exclude the maximum value
            ax.grid(True, which='both', color='white', linestyle='-', linewidth=0.5)

            # Set tick labels using the specified format
            ax.set_yticklabels([f"$\mathrm{{IR_{{i}}}} = {j}$" for j,_ in enumerate(y_ticks[:-1], start=1)], fontdict={'fontname': 'serif', 'fontsize': 10})


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
            ax.tick_params(axis='both', which='major', labelsize=8, labelcolor='black')
            
            # Set tick labels using the specified format
            ax.set_xticklabels([f"${{\\mathrm{{{int(x)}}}}}$" for x in ax.get_xticks()], fontdict={'fontsize': 10})

            # Show the x-axis label on every subplot
            ax.set_xlabel(r"$\mathrm{Generation}$", fontsize=12)
            ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)

        # Update the plot and process the event loop
        # plt.show(block=False)
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Save the figure in the specified formats
        for fmt in gp.config['runcontrol']['plot']['format']:
            fig.savefig(gp.config['runcontrol']['plot']['folder'] + f"{gp.runname}_RankallVsGeneration.{fmt}", dpi=300, format=fmt)

        # plt.pause(0.1)
        plt.close(fig)
