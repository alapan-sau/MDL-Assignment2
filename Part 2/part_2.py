
# Action Defintions

ACTION_UP = 'UP'
ACTION_LEFT = 'LEFT'
ACTION_DOWN = 'DOWN'
ACTION_RIGHT = 'RIGHT'
ACTION_STAY = 'STAY'
ACTION_SHOOT = 'SHOOT'
ACTION_HIT = 'HIT'
ACTION_CRAFT = 'CRAFT'
ACTION_GATHER = 'GATHER'
ACTION_NONE = 'NONE'

ACTIONS = [ACTION_UP, ACTION_LEFT, ACTION_DOWN, ACTION_RIGHT, ACTION_STAY, ACTION_SHOOT, ACTION_HIT, ACTION_CRAFT, ACTION_GATHER, ACTION_NONE]


# MM State Definitions

STATE_DORMANT = 'D'
STATE_READY = 'R'

MM_STATES = [STATE_DORMANT, STATE_READY]


# Position Definitions

POSITION_W = 'W'
POSITION_N = 'N'
POSITION_E = 'E'
POSITION_S = 'S'
POSITION_C = 'C'

POSITIONS = [POSITION_W , POSITION_N, POSITION_E, POSITION_S, POSITION_C]

# Numerical Values

ARROWS = [0,1,2,3]
MATS = [0,1,2]
HEALTHS = [0,25,50,75,100]


# OPERATION RESULTS
GO_LEFT = {
    'C' : 'W',
    'E' : 'C',
}

GO_STAY = {
    'C':'C',
    'W':'W',
    'E':'E',
    'S':'S',
    'N':'N',
}

GO_RIGHT = {
    'W':'C',
    'C':'E'
}

GO_UP ={
    'C':'N',
    'S':'C'
}

GO_DOWN = {
    'C':'S',
    'N':'C'
}

GO_UNSUCCESS = {
    'C':'E',
    'N':'E',
    'S':'E',

    ### will never happen
    'W':'W',
    'E':'E',
}

# Action Success Probability

SHOOT_PROB_SUCCESS = {
    'C':0.5,
    'W':0.25,
    'E':0.9,
}

MOVEMENT_PROB= {
    'C':0.85,
    'E':1,
    'W':1,
    'N':0.85,
    'S':0.85
}

HIT_PROB_SUCCESS = {
    'C':0.1,
    'E':0.2,
}


step_cost = -5

ALL_STATES = []

gamma = 0.999
error = 0.001

# Check if an action is valid or not in a given state
def isValidAction(state, action):

    position, mat, arrow, mm_state, health = state

    if health == 0:
        return action == ACTION_NONE

    if action == ACTION_NONE:
        return health == 0

    if action == ACTION_STAY:
        return True

    if action == ACTION_CRAFT:
        return position == POSITION_N and mat > 0

    if action == ACTION_GATHER:
        return position == POSITION_S

    if action == ACTION_SHOOT:
        return (position == POSITION_C or position == POSITION_E or position == POSITION_W) and arrow > 0

    if action == ACTION_HIT:
        return (position == POSITION_C or position == POSITION_E)

    if action == ACTION_UP:
        return (position == POSITION_S or position == POSITION_C)

    if action == ACTION_DOWN:
        return (position == POSITION_N or position == POSITION_C)
    
    if action == ACTION_LEFT:
        return (position == POSITION_E or position == POSITION_C)
    
    if action == ACTION_RIGHT:
        return (position == POSITION_W or position == POSITION_C)


# Gives next state, probability and rewards for a given state and action    
def transformation(state, action):
    position, mat, arrow, mm_state, health = state

    probability=[]
    new_states=[]
    rewards = []

    if(action == ACTION_NONE):
        return [1], [state], [-step_cost]
        
    elif(action == ACTION_LEFT):
        # all other things remain same, assuming mm_state is handled
        ### sucess
        new_position = GO_LEFT[position]
        prob = 1* MOVEMENT_PROB[position]

        new_states.append((new_position, mat, arrow, mm_state, health))
        probability.append(prob)

        ### failure
        new_position = GO_UNSUCCESS[position]
        prob = 1* (1-MOVEMENT_PROB[position])

        if(prob):
            new_states.append((new_position, mat, arrow, mm_state, health))
            probability.append(prob)

    elif(action == ACTION_DOWN):
        # all other things remain same, assuming mm_state is handled
        ### sucess
        new_position = GO_DOWN[position]
        prob = 1* MOVEMENT_PROB[position]

        new_states.append((new_position, mat, arrow, mm_state, health))
        probability.append(prob)

        ### failure
        new_position = GO_UNSUCCESS[position]
        prob = 1* (1-MOVEMENT_PROB[position])
    
        if(prob):
            new_states.append((new_position, mat, arrow, mm_state, health))
            probability.append(prob)

    elif(action == ACTION_RIGHT):
        # all other things remain same, assuming mm_state is handled
        ### sucess
        new_position = GO_RIGHT[position]
        prob = 1* MOVEMENT_PROB[position]

        new_states.append((new_position, mat, arrow, mm_state, health))
        probability.append(prob)

        ### failure
        new_position = GO_UNSUCCESS[position]
        prob = 1* (1-MOVEMENT_PROB[position])

        if(prob):
            new_states.append((new_position, mat, arrow, mm_state, health))
            probability.append(prob)

    elif(action == ACTION_UP):
        # all other things remain same, assuming mm_state is handled
        ### sucess
        new_position = GO_UP[position]
        prob = 1* MOVEMENT_PROB[position]

        new_states.append((new_position, mat, arrow, mm_state, health))
        probability.append(prob)

        ### failure
        new_position = GO_UNSUCCESS[position]
        prob = 1* (1-MOVEMENT_PROB[position])

        if(prob):
            new_states.append((new_position, mat, arrow, mm_state, health))
            probability.append(prob)
    
    elif (action == ACTION_STAY):
        # all other things remain same, assuming mm_state is handled
        ### sucess
        new_position = GO_STAY[position]
        prob = 1* MOVEMENT_PROB[position]

        new_states.append((new_position, mat, arrow, mm_state, health))
        probability.append(prob)

        ### failure
        new_position = GO_UNSUCCESS[position]
        prob = 1* (1-MOVEMENT_PROB[position])

        if(prob):
            new_states.append((new_position, mat, arrow, mm_state, health))
            probability.append(prob)
    
    elif(action == ACTION_SHOOT):
        ### success
        new_arrow = arrow -1
        new_health = health - 25
        prob = 1 * SHOOT_PROB_SUCCESS[position]
        
        new_states.append((position, mat, new_arrow, mm_state, new_health))
        probability.append(prob)

        ### failure
        new_arrow = arrow - 1
        new_health = health
        prob = 1 * (1-SHOOT_PROB_SUCCESS[position])
        
        new_states.append((position, mat, new_arrow, mm_state, new_health))
        probability.append(prob)          
    
    elif(action == ACTION_HIT):

        ### success
        new_states.append((position,mat, arrow, mm_state, max(0, health-50)))
        probability.append(HIT_PROB_SUCCESS[position])

        ### Failure
        new_states.append((position,mat, arrow, mm_state, health))
        probability.append(1 - HIT_PROB_SUCCESS[position])
        
    elif(action == ACTION_CRAFT):

        prob = 0
        prob_val = [0.5, 0.35, 0.15]
        
        for i in range(arrow+1,3):
            prob += prob_val[i-1-arrow]
            new_states.append((position, mat-1, i, mm_state, health));
            probability.append(prob_val[i-1-arrow])

        new_states.append((position, mat-1, 3, mm_state, health))
        probability.append(1-prob)

    elif(action == ACTION_GATHER):

        if mat == 2:
            new_states.append((position, 2, arrow, mm_state, health))
            probability.append(1)
        else:
            new_states.append((position, mat+1, arrow, mm_state, health))
            probability.append(0.75)
            new_states.append((position, mat, arrow, mm_state, health))
            probability.append(0.25)

    
    l = len(new_states)
    for i in range(0,l):
        if(new_states[i][4] == 0):
            rewards.append(50)
        else:
            rewards.append(0)

    for i in range(0, l):
        if mm_state == STATE_DORMANT:
            probability.append(probability[i]*0.2)
            probability[i] *= 0.8
            new_states.append((new_states[i][0], new_states[i][1], new_states[i][2], STATE_READY, new_states[i][4]))
            rewards.append(rewards[i])
        
        else:
            probability[i] *= 0.5 # Not attacking

            # Attacking but no effect except change in mm_state
            if position != POSITION_C and position != POSITION_E:
                probability.append(probability[i])
                new_states.append((new_states[i][0], new_states[i][1], new_states[i][2], STATE_DORMANT, new_states[i][4]))
                rewards.append(rewards[i])

    # Attack and state becomes original (i.e, IJ is unsuccessful)
    if mm_state == STATE_READY and (position == POSITION_C or position == POSITION_E):
        probability.append(0.5)
        new_states.append((position, mat, 0, STATE_DORMANT, min(health+25, 100)))
        rewards.append(-40)

    return probability, new_states, rewards

def vai():
    
    iteration = 0
    beta = 1
    utilities = []
    initital_utility = []
    for i in range(0, len(ALL_STATES)):
        initital_utility.append((0, ACTION_NONE))

    utilities.append(initital_utility)


    while(beta > error):

        beta = 0
        current_iteration_utilities = []

        policy = []

        for state in ALL_STATES:

            self_index = ALL_STATES.index(state)

            action_utilities = []

            for action in ACTIONS:
                if(isValidAction(state, action)):
                    probability, new_states, rewards = transformation(state, action)

                    sigma = 0
                    for i in range(0,len(probability)):

                        neighbour = new_states[i]
                        neighbour_index = ALL_STATES.index(neighbour)
                        sigma += probability[i] * (utilities[iteration][neighbour_index][0] * gamma + rewards[i])

                    action_utilities.append((step_cost + sigma,action))

            best_action_utility = -1000000000
            best_action = ''
            
            for action_utility, action in action_utilities:
                best_action_utility = max(action_utility, best_action_utility)
                if best_action_utility == action_utility:
                    best_action = action

            current_iteration_utilities.append((best_action_utility, best_action))

        utilities.append(current_iteration_utilities)

        for i in range(0,len(ALL_STATES)):
            beta = max( beta , abs(utilities[iteration+1][i][0] -  utilities[iteration][i][0]))
        iteration+=1

    return utilities

def create_trace(utilities):

    f = open('outputs/part_2_task_2.2_trace.txt', "w")

    for i in range(1, len(utilities)):
        f.write("\n\nIteration %d\n"%(i-1))
        for j in range(0, len(ALL_STATES)):
            f.write("(%1s, %1d, %1d, %1s, %3d) : %7s = [%5.3f]\n"%(ALL_STATES[j][0],ALL_STATES[j][1],ALL_STATES[j][2],ALL_STATES[j][3],ALL_STATES[j][4],utilities[i][j][1],utilities[i][j][0])  )

        
    f.close()


## Create ALL_STATES
for i in POSITIONS:
    for j in MATS:
        for k in ARROWS:
            for l in MM_STATES:
                for m in HEALTHS:
                    state = (i,j,k,l,m)
                    ALL_STATES.append(state)

utilities = (vai())
create_trace(utilities)


