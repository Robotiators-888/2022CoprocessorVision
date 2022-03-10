# setup startup script
echo "setting up startup script, wait for password prompt"
sudo cp findBall.service /etc/systemd/system/
sudo systemctl enable findBall.service
sudo systemctl start findBall.service