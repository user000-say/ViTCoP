"""Completed-block budget accounting and bounded schedule construction."""
import copy


def work(depth,initial,events):
    """Match LLaVA e018_contract.token_work: prune after counting that layer at pre-prune N."""
    if not isinstance(depth,int) or depth<1 or not isinstance(initial,int) or initial<1:
        raise ValueError('invalid depth/initial count')
    schedule={}
    prev_layer,prev_k=-1,initial
    for event in events:
        d,k=event['after'],event['keep']
        if not isinstance(d,int) or not isinstance(k,int) or not 0<=d<depth or d<=prev_layer or not 0<k<prev_k:
            raise ValueError('events must be valid post-block indices and decreasing positive K')
        schedule[d]=k; prev_layer,prev_k=d,k
    n,total=initial,0
    for layer in range(depth):
        total+=n
        if layer in schedule:
            n=schedule[layer]
    return total


def _two_match(depth,initial,early,first,last):
    target=work(depth,initial,early)
    final=early[-1]['keep']
    if not 1<=first<last<depth:return None
    # e018 token_work: initial*(first+1) + mid*(last-first) + final*(depth-1-last)
    denom=last-first
    mid=(target-initial*(first+1)-final*(depth-1-last))//denom
    if not final<mid<initial:return None
    result=[{'after':first,'keep':int(mid)},{'after':last,'keep':final}]
    if work(depth,initial,result)!=target:return None
    return result


def variants(depth,initial,known):
    required={'ONE','EARLY','LATE'}
    if set(known)!=required:raise ValueError('export actual ONE/EARLY/LATE schedules')
    for events in known.values():work(depth,initial,events)
    early=known['EARLY']; late=known['LATE']
    if len(early)!=2 or len(late)!=2:
        raise ValueError('this matrix expects two actual pruning events in EARLY/LATE')
    if early[-1]['keep']!=late[-1]['keep']:
        raise ValueError('EARLY/LATE final budgets differ')
    if early[0]!=late[0]:
        raise ValueError('EARLY/LATE must share first event for this focused matrix')
    if early[-1]['after']>=late[-1]['after']:
        raise ValueError('LATE final event must be later')
    output=copy.deepcopy(known); skipped=[]
    first,last=early[0]['after'],early[1]['after']
    candidate=_two_match(depth,initial,early,first,late[-1]['after'])
    if candidate:output['LATE_WMATCH']=candidate
    else:skipped.append('LATE_WMATCH infeasible with same first boundary')
    first2=max(1,first-2)
    if first2==first:first2=first+2
    candidate=_two_match(depth,initial,early,first2,last)
    if candidate:output['FIRST_SHIFT_WMATCH']=candidate
    else:skipped.append('FIRST_SHIFT_WMATCH infeasible')
    middle=(first+last)//2; target=work(depth,initial,early)
    best=None; rank=None
    if first<middle<last:
        for high in range(early[0]['keep']+1,initial):
            for low in range(early[-1]['keep']+1,min(high,early[0]['keep'])):
                events=[{'after':first,'keep':high},{'after':middle,'keep':low},
                        {'after':last,'keep':early[-1]['keep']}]
                used=work(depth,initial,events)
                if used<=target:
                    order=(target-used,abs(high-early[0]['keep'])+abs(low-early[0]['keep']))
                    if rank is None or order<rank:rank,best=order,events
    if best:output['THREE_WMATCH']=best
    else:skipped.append('THREE_WMATCH infeasible')
    annotations={}
    for name,events in output.items():
        used=work(depth,initial,events)
        annotations[name]={'visual_token_layers':used,'difference_from_EARLY':used-target,
                           'relative_difference':(used-target)/target,
                           'near_work_matched':abs(used-target)/target<=.005}
    return output,annotations,skipped
