def ways_to_go_here(roads, city_from, city_to):
    if city_to == city_from:
        return 1
    summ_of_ways = 0
    for city1, city2 in roads:
        if city2==city_to:
            summ_of_ways += ways_to_go_here(roads, city_from, city_to=city1)
    return summ_of_ways


N, k = map(int, input("Сколько городов и дорог? ").split())
roads = []
for i in range(k):
    road_from, road_to = map(int, input(f"Дорога {i}: ").split())
    roads.append((road_from, road_to))
city_from, city_to = map(int, input("Откуда и куда?").split())
# must_to_visit = list(map(int, input("Куда обязательно заехать? ").split()))
# forbidden_to_visit = list(map(int, input("Куда нельзя? ").split()))
# one_way = bool(input("Ориентированный? "))
print(ways_to_go_here(roads, city_from, city_to))