from urllib.parse import unquote

encoded_key = "0JHLxzLbTMA6JAxou0oL5T2rYLYO1kezLVjW3T53n%2BDcd4Q4fPdYFASUz90F%2FaPrWW8M7lUvCkP7t9TpSojx%2FQ%3D%3D"
decoded_key = unquote(encoded_key)
print(decoded_key)