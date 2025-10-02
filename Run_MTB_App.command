
#!/bin/zsh
PROJECT_DIR="/Users/simonlay/Downloads/mtb_agent"
CONDA_INIT="$HOME/opt/anaconda3/etc/profile.d/conda.sh"

osascript <<EOF
tell application "Terminal"
    do script "source $CONDA_INIT && conda activate mtb && cd '$PROJECT_DIR' && python -m streamlit run app_streamlit.py"
    activate
end tell
EOF