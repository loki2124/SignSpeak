mkdir -p ~/.streamlit/
echo "
[general]n
email = “afiak@andrew.cmu.edu”
" > ~/.streamlit/credentials.toml
echo "
[server]n
headless = truer
enableCORS=falsen
port = 8051
" > ~/.streamlit/config.toml

