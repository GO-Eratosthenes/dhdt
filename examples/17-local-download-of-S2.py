import pystac
import satsearch
import stac2dcache

utm_zone = "5"
latitude_band = "V"
grid_square = "MG"

api_url = "https://earth-search.aws.element84.com/v0"

collection_id = "sentinel-s2-l1c"

search_kwargs = dict(
    url=api_url,
    collections=[collection_id],
    query=[
        f"sentinel:utm_zone={utm_zone}",
        f"sentinel:latitude_band={latitude_band}",
        f"sentinel:grid_square={grid_square}"
    ]
)
search = satsearch.Search.search(**search_kwargs)

items_l1c = search.items()
#print(items_l1c.summary(params=["date", "id", "sentinel:data_coverage", "eo:cloud_cover"]))

catalog_id = "red-glacier_sentinel-2"

catalog = pystac.Catalog(
    id=catalog_id,
    description='This catalog contains Sentinel-2 tiles for the Red Glacier (Alaska)'
)
# add search results to catalog
for item_collection in items_l1c:
    items = (pystac.Item.from_dict(item._data) for item in item_collection)
    catalog.add_items(items)

# replace the self-links of the items to remote
# with relative links
catalog.normalize_hrefs(catalog_id)

template = "${collection}/${year}/${month}/${day}"
catalog.generate_subcatalogs(template)
