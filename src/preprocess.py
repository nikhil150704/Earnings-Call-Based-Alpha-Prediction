if __name__ == "__main__":
  prepare_text_for_nlp("data/raw/INFY_Q1_July_22.pdf", "current")
  prepare_text_for_nlp("data/raw/INFY_Q2_October_21.pdf", "prev1")
  prepare_text_for_nlp("data/raw/INFY_Q3_January_22.pdf", "prev2")
  prepare_text_for_nlp("data/raw/INFY_Q4_April_22.pdf", "prev3")
