for session in $(tmux list-sessions -F "#{session_id}"); do
  if [[ "$session" != "0" && "$session" != "1" && "$session" != "2" && "$session" != "16" ]]; then
    tmux kill-session -t "$session"
  fi
done