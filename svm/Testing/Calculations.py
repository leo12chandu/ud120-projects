import math

entropy_parent = 1.0

#print math.log((2./3), 2)
entropy_3_4_children = (-(2./3) * math.log(2./3, 2) - (1./3) * math.log(1./3, 2))

print 'entropy_3_4_children = ', entropy_3_4_children

entropy_1_4_children = (1./1) * math.log(1./1, 2)

print 'entropy_1_4_children = ', entropy_1_4_children

entropy_children_weighted = (3./4) * entropy_3_4_children + (1./4) * entropy_1_4_children
print 'entropy_children_weighted = ', entropy_children_weighted

information_gain = entropy_parent - entropy_children_weighted

print 'information_gain = ', information_gain
