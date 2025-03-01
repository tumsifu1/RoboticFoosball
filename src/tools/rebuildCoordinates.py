def rebuild_absolute_coordinates(region_row, region_col, local_x, local_y, region_width, region_height):
    absolute_x = region_col * region_width + local_x
    absolute_y = region_row * region_height + local_y
    return absolute_x, absolute_y