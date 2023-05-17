
from matplotlib import markers, pyplot as plt
import seaborn as sns
import pandas as pd
import sys

x_data = [0, 20, 40, 60, 80, 100]

# bleu scores for percentage shuffle of test corpus
en_fr_ns_bleu = [49.4, 35.3, 22.8, 15.1, 11.7, 10.7]

en_fr_s_bleu = [41.0, 41.0, 41.0, 40.9, 41.0, 41.0]

fr_en_ns_bleu = [51.4, 35.2, 21.6, 13.1, 9.2, 8.2]

fr_en_s_bleu = [43.4, 43.4, 43.3, 43.3, 43.2, 43.2]

en_es_ns_bleu = [60.1, 42.9, 27.0, 17.9, 13.2, 11.5]

en_es_s_bleu = [50.0, 50.0, 50.0, 49.9, 50.0, 49.9]

es_en_ns_bleu = [60.0, 39.4, 22.9, 13.5, 9.0, 7.9]

es_en_s_bleu = [50.4, 50.3, 50.2, 50.1, 50.1, 50.0]

es_fr_ns_bleu = [48.6, 35.1, 22.5, 14.9, 10.9, 9.8]

es_fr_s_bleu = [42.1, 42.1, 42.1, 41.9, 42.0, 42.0]

fr_es_ns_bleu = [49.0, 37.1, 24.8, 17.2, 13.0, 12.0]

fr_es_s_bleu = [43.6, 43.7, 43.6, 43.6, 43.5, 43.6]


# ribes scores for percentage shuffle of test corpus
en_fr_ns_ribes = [0.4191946098, 0.3430630793694428, 0.2676072983858784,
            0.22297443234384137, 0.20103510091920743, 0.1933731008370864]

en_fr_s_ribes = [0.3705725944, 0.3699020588981276, 0.3703300447074176,
           0.3697306502643233, 0.37000500912040174, 0.3696384125150498]

fr_en_ns_ribes = [0.4326911686, 0.3419353748453054, 0.2540607638102149,
            0.20123613167641966, 0.17193849897659905, 0.15128975983908535]


fr_en_s_ribes = [0.3808773224, 0.37977109926040603, 0.37990541565973,
           0.38024291359015194, 0.3802265900965013, 0.3801530457493345]

en_es_ns_ribes = [0.4666398478, 0.35936391802089884, 0.2691112824513244,
            0.21436066773507548, 0.18401995651146122, 0.17893349520504573]

en_es_s_ribes = [0.39883684, 0.3989862248023525, 0.3987327024672647,
           0.3984914253427039, 0.3979559777351164, 0.398689080030]

es_en_ns_ribes = [0.4785935565, 0.3439467532259099, 0.2452203378608851, 
            0.1918854194823256, 0.15816945555455642, 0.14224144506878383]

es_en_s_ribes = [0.4129865422, 0.4136030636245803, 0.41274772763474066, 
            0.4124580001259575, 0.4121875600490343, 0.41249582216460084]

es_fr_ns_ribes = [0.4223695544, 0.3481257657782087, 0.2711937304642771, 
            0.22271304372028297, 0.19565577964164507, 0.1792023360136034]

es_fr_s_ribes = [0.3767520137, 0.3772696863468623, 0.3769253767485502, 
            0.3777655155957791, 0.3769461667376226, 0.37782475466409554]

fr_es_ns_ribes = [0.4109877711, 0.34402997564602766, 0.27535982210350085, 
            0.22798144634627576, 0.20167449390518266, 0.19671850958781434]

fr_es_s_ribes = [0.3753351217, 0.37545075504565545, 0.37557214030614616,
            0.3753783639029357, 0.37561749104975395, 0.37587845320049285]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please enter either bleu or ribes")

    evaluation = sys.argv[1]

    if evaluation == "bleu":

        data_preproc = pd.DataFrame({
            'Shuffling Percentage': x_data,
            'en-fr orig.': en_fr_ns_bleu,
            'en-fr shuf.': en_fr_s_bleu,
            'fr-en orig.': fr_en_ns_bleu,
            'fr-en shuf.': fr_en_s_bleu,
            'en-es orig.': en_es_ns_bleu,
            'en-es shuf.': en_es_s_bleu,
            'es-en orig.': es_en_ns_bleu,
            'es-en shuf.': es_en_s_bleu,
            'es-fr orig.': es_fr_ns_bleu,
            'es-fr shuf.': es_fr_s_bleu,
            'fr-es orig.': fr_es_ns_bleu,
            'fr-es shuf.': fr_es_s_bleu
            })
        
        dfm = data_preproc.melt('Shuffling Percentage',
                            var_name='Model', value_name='BLEU Score')
    
    else:
        data_preproc = pd.DataFrame({
            'Shuffling Percentage': x_data,
            'en-fr orig.': en_fr_ns_ribes,
            'en-fr shuf.': en_fr_s_ribes,
            'fr-en orig.': fr_en_ns_ribes,
            'fr-en shuf.': fr_en_s_ribes,
            'en-es orig.': en_es_ns_ribes,
            'en-es shuf.': en_es_s_ribes,
            'es-en orig.': es_en_ns_ribes,
            'es-en shuf.': es_en_s_ribes,
            'es-fr orig.': es_fr_ns_ribes,
            'es-fr shuf.': es_fr_s_ribes,
            'fr-es orig.': fr_es_ns_ribes,
            'fr-es shuf.': fr_es_s_ribes
            })

        dfm = data_preproc.melt('Shuffling Percentage',
                            var_name='Model', value_name='RIBES Score')

    sns.set_theme(style="darkgrid")
    markers_list = ['o', 'o', 'x', 'x', 's', 's', '+', '+', '*', '*', 11, 11]

    sns.set(rc = {'figure.figsize':(12,9), 'lines.linewidth':0.7, 'legend.framealpha': 0.5})
    sns.set(font_scale=1.38)

    palette = {
        'en-fr orig.': '#12ee85',
        'en-fr shuf.': '#9a5d8e',
        'fr-en orig.': '#f99529',
        'fr-en shuf.': '#eaec8f',
        'en-es orig.': '#214c54',
        'en-es shuf.': '#937860',
        'es-en orig.': '#265bc9',
        'es-en shuf.': '#dc37e6',
        'es-fr orig.': '#51a6d4',
        'es-fr shuf.': '#734da6',
        'fr-es orig.': "#576965",
        'fr-es shuf.': "#8f4904"
        }

    pal = sns.color_palette(None, 12)
    print(pal.as_hex())

    if evaluation == "bleu":
        ax = sns.pointplot(x='Shuffling Percentage', y='BLEU Score', hue='Model', 
                    data=dfm, scale=1, markers=markers_list, palette=palette)
        
        ax.set_ylabel('BLEU Score',fontsize=20)

    else:
        ax = sns.pointplot(x='Shuffling Percentage', y='RIBES Score', hue='Model', 
                    data=dfm, scale=1, markers=markers_list, palette=palette)
        
        ax.set_ylabel('RIBES Score',fontsize=20)

    ax.get_legend().set_title(None)
    plt.setp(ax.collections, sizes=[200])
    ax.set_xlabel('Shuffling Percentage',fontsize=20)
    print(data_preproc)
    print(dfm)

    if evaluation == "bleu":
        plt.savefig('percentage_shuffle.svg',
                    format='svg', dpi=1200, bbox_inches='tight')
    
    else:
        plt.savefig('percentage_shuffle_RIBES.svg',
            format='svg', dpi=1200, bbox_inches='tight')
