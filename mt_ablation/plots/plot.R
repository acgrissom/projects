library(tidyverse)
#library(hrbrthemes)
library(ggplot2)
df  <- read.csv("new.csv")

theme_set(
    #theme_minimal() +
    theme(legend.position="bottom",
                                  legend.title=element_blank(),
                                  text = element_text(size=8),
                                  #plot.background = element_rect(
                                  #    fill = "white",
                                  #    color = "white"
          )
)


#define plotting function
plot_faceted <- function(df, width=8, height=4, filename="facet_plot.svg") {
    plot <- ggplot(df, aes(x=percent_perturbed,
                           y=bleu_score,
                           color=case,
                           shape=case,
                           #linetype=shuffled
                           )) +
        labs(y = "BLEU", x = "Percent Perturbed") +
        geom_line() +
        geom_point(size=1) +
        facet_grid(model_type ~ perturbation)
        #scale_color_brewer(palette = "Dark2")
    ggsave(filename, plot=plot, width=width,height=height)
    ggsave(paste(filename,".png", sep=""), plot=plot, width=width,height=height)
    
} 


#filter by model type
unshuffled_df <- df[grep("non-shuffled", df$shuffled),]
shuffled_df <- df[grep("^shuffled$", df$shuffled),]

width = 8
height = 4
plot_faceted(unshuffled_df, width=width, height=height, filename="unshuffled.svg")
plot_faceted(shuffled_df, width=width, height=height, filename="shuffled.svg")
#remove all rows except those that match the regex for model
                                        #unshuffled_df <- unshuffled_df[grep("*en-zh*", unshuffled_df$model_type),












