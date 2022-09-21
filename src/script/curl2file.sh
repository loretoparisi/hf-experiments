function curl2file() {
    $(curl -o /dev/null -L --head -w '%{url_effective}' $1 2>/dev/null | tail -n1) ; curl -O $1
}
curl2file $@