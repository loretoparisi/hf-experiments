function curl2redirect() {
    X=$(curl -o /dev/null -L --head -w '%{url_effective}' $1  2>/dev/null) ; curl "$X" -o $2
}
curl2redirect $@