# E-commerce Assistant

An intelligent e-commerce assistant that recommends UK products and remembers past queries.


## Overview
- **Offline Product Dataset**: Stores product `name`, `category`, and `price`.

- **Query Matching**:
  - Checks if user input contains a `category` or product `name`.
  - If matches `found` → suggests matched products.
  - Otherwise → provides top `default` products.

- **Memory-aware Responses**:
  - Stores all `user-bot` interactions.
  - References `past` queries in current suggestions.

- **Streamlit UI**: Handles input, output, and displays conversation history.


## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py

### Sample Questions to Try

### Do you have any electronics?
### Show me some clothing items.
### I need some footwear recommendations.

### Tell me about Gaming Laptop.
### Do you sell Organic Tea Set?

### What did you suggest for electronics last time?
### Any more clothing options?

### I'm not sure what to buy.
### What are your best offers today?

### Do you have eco-friendly products?
### Show me cheap options under £50.
```