# ===============================================
# Fill in the following fields
# ===============================================
# Title:
# Description:
# Author: 
# Date:


# ===============================================
# Packages
# ===============================================
library(tidyverse)
library(tidytext)


# ===============================================
# Import data
# ===============================================
# for demo purposes of the "template", we use data starwars
# (but you will have to replace this with the data in "simpsons-transcripts.txt")
dat <- read.delim("simpsons-transcripts.txt", sep = "^")


# ===============================================
# Define "ui" for application
# ===============================================

ui <- fluidPage(
  
  titlePanel("Simpsons Text Analysis"),
  fluidRow(
    # replace with your widgets
    column(3,
           
           radioButtons(inputId = "stopwords", 
                        label = "Remove Stopwords", 
                        choices = c("Remove" = "TRUE",
                                    "Do Not Remove" = "FALSE"), 
                        selected = "TRUE")
    ),
    
    column(3,
         
           sliderInput(inputId = "numberwords",
                       label = "Top Number of Words",
                       min = 1,
                       max = 30,
                       value = 15)
    ),
    
    # replace with your widgets
    column(3,
          
           sliderInput(inputId = "season", 
                        label = "Seasons to Include", 
                        min = 1,
                        max = 33,
                       value = c(1, 33))
    ),
    
    
    column(3,
           
           sliderInput(inputId = "numberbigrams",
                       label = "Minimum Bigram Count",
                       min = 50,
                       max = 400,
                       value = 100)
    ),
    

  
    # replace with your widgets
  
  
  hr(),
  
  tabsetPanel(type = "tabs",
              tabPanel("Analysis1",
                       h3("Word Frequency Analysis"),
                       plotOutput("plot1"),
                       hr(),
                       h4('Word Count Table'),
                       dataTableOutput('table1')),
              tabPanel("Analysis2", 
                       h3("Bigram Analysis"),
                       plotOutput("plot2"),
                       hr(),
                       h4('Bigram Word Count Table'),
                       dataTableOutput('table2'))
  )
)
)


# ===============================================
# Define Server "server" logic
# ===============================================

server <- function(input, output) {
  
  # you may need to create reactive objects
  # (e.g. data frame to be used in plot1)
  
  library(tidytext)
  library(igraph)       
  library(ggraph)
  

  
  dat_freq <- reactive({
  
    lower = input$season[1]
    higher = input$season[2]
    dat = dat[dat$season >= lower & dat$season <= higher,]
    dat_tokens = unnest_tokens(tbl = dat, output = word, input = text)
    
    
    if (input$stopwords == TRUE) {
      tidy_dat = dat_tokens %>% 
        anti_join(stop_words, by = "word") %>%
        count(word)
    
        dat_top_words  = tidy_dat %>%
          arrange(desc(n)) 
      
      
    } else {
      dat_freqs = dat_tokens %>% count(word)
      
      dat_top_words = dat_freqs %>% 
          arrange(desc(n))
        
      
    }
    dat_top_words
  })
  
  bigrams_freq = reactive({
    lower = input$season[1]
    higher = input$season[2]
    dat = dat[dat$season >= lower & dat$season <= higher,]
    
    bigrams <- dat %>%
      unnest_tokens(output = bigram, input = text, token = "ngrams", n = 2) %>%
      filter(!is.na(bigram))
    
    bigrams_separated <- bigrams %>%
      separate(bigram, c("word1", "word2"), sep = " ")
    
    bigrams_filtered <- bigrams_separated %>%
      filter(!word1 %in% stop_words$word) %>%
      filter(!word2 %in% stop_words$word)
    
    count_bigrams <- bigrams_filtered %>%
      count(word1, word2, sort = TRUE)
    
    count_bigrams
  })
  
  
  # ===============================================
  # Outputs for the first TAB
  # ===============================================
  
  # code for plot1
  output$plot1 <- renderPlot({
    # replace the code below with your code!!!
    getdat = dat_freq()
    ourdat = slice_head(getdat, n = input$numberwords)
    
    ggplot(data = ourdat,
           aes(x = reorder(word, -n), y = n)) +
      geom_col(fill = "steelblue") + 
      labs(title = "Top Frequent Words") +
      xlab("word") +
      ylab("count")
  })
  
  # code for numeric summaries of frequencies
  output$table1 <- renderDataTable({
    # replace the code below with your code!!!
    dat_freq()
  })
  
  
  # ===============================================
  # Outputs for the second TAB
  # ===============================================
  
  # code for plot2
  output$plot2 <- renderPlot({
    # replace the code below with your code!!!
    count_bigrams = bigrams_freq()
    bigrams_united = count_bigrams %>%
      unite(bigram, word1, word2, sep = " ")
    
    bigramsamount <- slice(bigrams_united, 1:input$numberwords)
    
    bigrams_graph <- count_bigrams %>%
      filter(n > input$numberbigrams) %>%
      graph_from_data_frame()
    
    ggraph(bigrams_graph, layout = "fr") +
      geom_edge_link() +
      geom_node_point() +
      geom_node_text(aes(label = name), vjust = 1, hjust = 1) +
      labs(title = "Common igrams in The Simpsons") +
      theme_void() 
  })
  
  # code for statistics
  output$table2 <- renderDataTable({
    # replace the code below with your code!!!
    bigrams_freq()
  })
  
}



# ===============================================
# Run the application
# ===============================================

shinyApp(ui = ui, server = server)

