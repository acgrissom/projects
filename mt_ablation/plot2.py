
from matplotlib import markers, pyplot as plt
import seaborn as sns
import sys
import pandas as pd

x_data = [0, 20, 40, 60, 80, 100]

en_ko_ns_bleu = [38.6, 27.4, 17.8, 12.1, 9.1, 8.3]
en_ko_s_bleu = [26.8, 26.7, 26.7, 26.6, 26.7, 26.6]
ko_en_ns_bleu = [34.7, 27.1, 18.9, 13.9, 11.2, 10.3]
ko_en_s_bleu = [27.8, 27.8, 27.6, 27.6, 27.5, 27.4]

en_ko_ns_ribes = [0.23515699277441263, 0.18641140502291872, 0.14011237905502327, 
            0.10951982963163583, 0.09354898730771907, 0.08933493406331364]

en_ko_s_ribes = [0.18521368685497644, 0.18514885911685233, 0.1845424894594723, 
           0.18444220102417425, 0.18415715239503583, 0.18376460242593237]

ko_en_ns_ribes = [0.28716229056009573, 0.25163613704637705, 0.20062906620822993, 
            0.16997396441635015, 0.15126837076815144, 0.14454462153724926]

ko_en_s_ribes = [0.2500557717703285, 0.24959250631589025, 0.24813053746691963, 
           0.24858086276376307, 0.2485260896786634, 0.2477164019996124]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please enter either bleu or ribes")

    evaluation = sys.argv[1]

    if evaluation == "bleu":

        data_preproc = pd.DataFrame({
            'Shuffling Percentage': x_data,
            'en-ko orig.': en_ko_ns_bleu,
            'en-ko shuf.': en_ko_s_bleu,
            'ko-en orig.': ko_en_ns_bleu,
            'ko-en shuf.': ko_en_s_bleu
            })
        
        dfm = data_preproc.melt('Shuffling Percentage',
                            var_name='Model', value_name='BLEU Score')
        
    else:

        data_preproc = pd.DataFrame({
            'Shuffling Percentage': x_data,
            'en-ko orig.': en_ko_ns_ribes,
            'en-ko shuf.': en_ko_s_ribes,
            'ko-en orig.': ko_en_ns_ribes,
            'ko-en shuf.': ko_en_s_ribes
            })
        
        dfm = data_preproc.melt('Shuffling Percentage',
                            var_name='Model', value_name='RIBES Score')
        
        
    sns.set_theme(style="darkgrid")
    markers_list = ['o', 'o', 'x', 'x']  # ['s', 's', '+', '+', '*', '*', 11, 11,]

    sns.set(rc = {'figure.figsize':(12,9), 'lines.linewidth':0.7, 'legend.framealpha': 0.5})
    sns.set(font_scale=1.38)

    palette = {
        'en-ko orig.': '#12ee85',
        'en-ko shuf.': '#9a5d8e',
        'ko-en orig.': '#f99529',
        'ko-en shuf.': '#eaec8f'
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
        plt.savefig('percentage_shuffle2.svg',
                format='svg', dpi=1200, bbox_inches='tight')
        
    else:
        plt.savefig('percentage_shuffle2_RIBES.svg',
                format='svg', dpi=1200, bbox_inches='tight')
