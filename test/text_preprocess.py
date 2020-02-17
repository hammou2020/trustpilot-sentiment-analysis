s = "Had to order a new ID tag for our golden retriever due to switching cell providers and getting new phone numbers. Boomerang provides great quality stainless tags in less than one weeks time!"
s = preprocess(s, False)
x = onehot_encode(s, max_tokens, all_chars)
assert s[:max_tokens] == onehot_decode(x, all_chars)