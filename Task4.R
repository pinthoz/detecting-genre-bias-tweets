library(arules)
library(tm)
library(textclean)
library(stringr)
library(quanteda)

data <- read.csv("data/EXIST2025_train.csv")

process_tweets <- function(tweets) {
  # Initial cleaning
  tweets <- replace_html(tweets)                             # expose ampersands (&) to be removed later
  tweets <- str_replace_all(tweets, "@\\w+", "USER")         # replace usernames with USER token
  tweets <- str_replace_all(tweets, "http\\S+", "URL")       # replace links with URL token
  tweets <- str_replace_all(tweets, "\\.(\\S)", ". \\1")     # add space after dots
  # Create corpus
  corpus <- corpus(tweets)
  # Tokenize with careful handling of important features
  toks <- tokens(
    corpus,
    remove_punct = TRUE,
    remove_symbols = TRUE,
    what = "word"
  )
  # Normalize text
  toks <- tokens_tolower(toks)
  toks <- tokens_remove(toks, stopwords("english"))
  toks <- tokens_remove(toks, pattern = "^\\d+$", valuetype = "regex")
  toks <- tokens_wordstem(toks)
  
  return(toks)
}
toks <- process_tweets(data$tweet)
cleaned_tweets <- sapply(toks, function(x) paste(x, collapse = " "))

# Tokenize tweets into words
itemList <- strsplit(cleaned_tweets, "\\s+")
itemList <- lapply(itemList, function(x) x[nchar(x) > 0])  # Remove empty tokens

# Now combine demographics and tweets WITHOUT double prefixing
data$transaction <- mapply(function(demo_row, tweet_tokens) {
  # Add prefixes here only
  demo_tokens <- paste0(c("Gender", "Age", "Ethnicity", "Education", "Country", "Label"), "=", demo_row)
  c(demo_tokens, tweet_tokens)
}, split(data[, c("gender", "age", "ethnicity", "education", "country", "label_task1_1")], seq_len(nrow(data))),
itemList,
SIMPLIFY = FALSE)

# Convert to transactions
trans <- as(data$transaction, "transactions")

# Run apriori
rules <- apriori(
  trans,
  parameter = list(supp = 0.005, conf = 0.6, minlen = 1),
  appearance = list(rhs = c("Label=YES", "Label=NO"), default = "lhs")
)

rules_sorted <- sort(rules, by = "lift", decreasing = TRUE)

# Inspect top 10 rules
arules::inspect(head(rules_sorted, 10))


# Columns of interest
demo_cols <- c("gender", "age", "ethnicity", "education", "country")

for (col in demo_cols) {
  data[[col]] <- paste0(toupper(substring(col, 1, 1)), substring(col, 2), "=", data[[col]])
}

# Extract unique tokens from these columns
demo_tokens <- unique(unlist(lapply(demo_cols, function(col) unique(data[[col]]))))

rules_demo <- subset(rules, 
                     sapply(LIST(lhs(rules)), function(items) {
                       any(items %in% demo_tokens)
                     })
)

# Subset rules where rhs = "Label=NO"
rules_no <- subset(rules_demo, rhs %in% "Label=NO")
rules_no<- sort(rules_no,by="lift",decreasing=TRUE)
arules::inspect(head(rules_no, 10))

# Inspect the top 10 rules sorted by lift
rules_sorted_by_lift <- sort(rules_demo, by = "lift", decreasing = TRUE)
arules::inspect(head(rules_sorted_by_lift, 10))

