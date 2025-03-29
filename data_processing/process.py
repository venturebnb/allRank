import numpy as np

def flatten_groups(grouped_rankings):
    ranking_groups = []

    for search_date, search_date_grouped_items in grouped_rankings.items():
        for _, check_in_grouped_items in search_date_grouped_items.items():
            for _, duration_grouped_items in check_in_grouped_items.items():
                for _, guest_count_grouped_items in duration_grouped_items.items():
                    ranking_groups.append(guest_count_grouped_items[:])

    return ranking_groups


def combine_data(ranking_groups, property_map, seed=42):
    combined_data = []
    np.random.seed(seed)

    for ranking_group in ranking_groups:
        group_data = []
        group_rankings = []
        listing_ids = []
        group_metadata = {}

        for ranking_item in ranking_group:
            ranking_attrs = ranking_item["scaled_attrs"]
            property_id = ranking_item["propertyId"]
            ranking = ranking_item["scaled_ranking"]

            property_obj = property_map[property_id]
            property_attrs = property_obj["attrs"]
            property_encoded_title = property_obj["encoded_title"]

            property_data = np.concatenate([
                property_attrs,
                property_encoded_title
            ])

            attrs = np.concatenate([ranking_attrs, property_data])

            airbnb_id = property_obj["airbnbId"]
            guesty_listing_id = property_obj["guestyListingId"]
            listing_ids.append({'airbnb_id': airbnb_id, 'guesty_listing_id': guesty_listing_id})

            group_data.append(attrs)
            group_rankings.append(ranking)

            group_metadata = {
                "durationOfStay": ranking_item["durationOfStay"],
                "searchDateOffset": ranking_item["searchDateOffset"],
                "checkInDateOffset": ranking_item["checkInDateOffset"],
                "guestCount": ranking_item["guestCount"]
            }

        group_data_np = np.array(group_data)
        group_rankings_np = np.array(group_rankings)

        _, unique_indices = np.unique(group_rankings_np[::-1], return_index=True)
        unique_group_data = group_data_np[unique_indices]
        unique_group_rankings = group_rankings_np[unique_indices]

        shuffled_indices = np.random.permutation(len(unique_group_data))
        shuffled_listing_ids = [listing_ids[id] for id in shuffled_indices]
        combined_data.append(
            (
                unique_group_data[shuffled_indices, :],
                unique_group_rankings[shuffled_indices],
                shuffled_listing_ids,
                group_metadata
            )
        )

    return combined_data

def get_datasets(grouped_rankings, property_map):
    ranking_groups = flatten_groups(grouped_rankings)
    combined_data = combine_data(ranking_groups, property_map)
    return combined_data

