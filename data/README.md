# SciQAG-24D Dataset

## Files Description

### final_all_select1000.json
- **Download:** [final_all_select1000.json](https://drive.google.com/file/d/1nUxRdC1F1e0Rb5BinHi4-y5_ce2UY1wn/view?usp=sharing).
- **Format:** List of dictionaries
- **Dictionary Keys:**
  - `doi`
  - `input`
  - `output`
  - `journal`
  - `keywords`
  - `Q&A`
  - `num_Q&A`
  - `txt`
  - `index`
  - `scores`
- **Notes:**
  - Some categories do not reach 1000 entries.
  - Total data volume: 22,743 entries.

### select_50.json
- Directly selected the top 50 entries from `final_all_select1000.json` for visualization examples.

### Train_qas_179511.json
- Data from `final_all_select1000.json` excluding the test set.
- Context independence removed using regular expressions.
- **Download:** [Train_qas_179511.json](https://drive.google.com/file/d/1i1DS7zMjWmW6JboB95ddn2rjNPG8Ot4f/view?usp=sharing).
- **Format:** List of dictionaries
  - Example:
    ```json
    {
      "instruction": "You are a helpful assistant to answer scientific questions. Add details to answers as much as possible, such as answer the specific chemical elements and numbers.",
      "input": "q",
      "output": "a"
    }
    ```

### Test_qas_8531.json
- Contains 1200 articles (24 categories * 50 entries each).
- Only retains Q&A pairs with scores >= 3 in all dimensions except context independence, which is removed using regular expressions.
- **Format:** List of dictionaries
  - Example:
    ```json
    {
      "Q": "q",
      "A": "a"
    }
    ```
