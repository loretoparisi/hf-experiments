function curl2contentlength() {
    curl -sI -L -H 'Accept-Encoding: gzip,deflate' $1 | grep -i Content-Length | awk 'END{print $2}'
}
curl2contentlength $@