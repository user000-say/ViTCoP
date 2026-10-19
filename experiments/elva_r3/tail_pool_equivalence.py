"""Conditional design check, not evidence about uninspected native COVER code."""
import itertools
import json


def verify():
    cases=0
    for order in itertools.permutations(range(8)):
        core=list(order[:6]);tail=core[-2:];outside=list(order[6:])
        priority={value:index for index,value in enumerate(order)}
        for count in range(3):
            for removed in itertools.combinations(tail,count):
                for added in itertools.combinations(outside,count):
                    pool=(set(core)-set(removed))|set(added)
                    assert sorted(pool,key=priority.get)[:2]==list(order[:2])
                    cases+=1
    return dict(exhaustive_small_cases=cases,legacy_pool_size=172,tail_capacity=32,
                protected_prefix=172-32,requested=60,requested_inside_protected_prefix=60<=172-32,
                condition="COVER only removes from the bottom b of score TopK and inserts lower-score outside candidates",
                conclusion="same-score top60 after that pool equals plain top60; inspect native code before applying this condition",
                pretrained_vlm_claim=False)


if __name__=='__main__':print(json.dumps(verify(),indent=2))
