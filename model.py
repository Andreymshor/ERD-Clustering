# make sure your labels in image are the same as the labels in here (along w capitalization)

# entity: looks like a rectangle
# weak entity: looks like 2 rectangles on top of each other
# rel: a relationship: looks like a diamond
# ident_rel: an identifying relationship: Looks like 2 diamonds on top of each other
# rel_attr: an attribute that belongs to a relationship, looks like an oval/eclipse
# many: appears at the end of a connection, has 3 lines coming out of it
# one: appears at the end of a connection, has 1 or 2 vertical lines like so: ||
# rel_type: used for connections that don't follow standard format. These connections look like so: "1..*", "*..1", etc
labels = ['entity', 'weak_entity', 'rel', 'ident_rel', 'rel_attr', 'many', 'one' , 'rel_type']