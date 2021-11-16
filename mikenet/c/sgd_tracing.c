// 1. Weight update in SGD

bptt_apply_deltas(net) // I think this is the function that updates the weights
    Net *net;
{
    Real *w, *g, e;   // w = weight, g = gradient, e = epsilon/learning rate
    unsigned char *f; // f = frozen
    Connections *c;
    int nc, i, j, x, nto, nfrom;

    nc = net->numConnections;
    for (x = 0; x < nc; x++) // Loop through every connections
    {
        c = net->connections[x];
        if (c->locked)
            continue;
        nto = c->to->numUnits;
        e = c->epsilon;
        for (i = 0; i < nto; i++)
        {
            w = c->weights[i];
            g = c->gradients[i];
            f = c->frozen[i];
            nfrom = c->numIncoming[i];
            for (j = 0; j < nfrom; j++)
            {
                if (*f++ == 0) // If not frozen
                {
                    *w -= *g * e; // Update weight
                }
                *g++ = 0;
                w++;
            }
        }
    }
    return 0;
}

// 2. Getting the gradient

int bptt_compute_gradients(Net *net, Example *ex)
{
    int ng, nt, nc, t, i, g, nu, c, rc; // ng: number of groups, nt: number of time steps, nc: number of connections,
    // t: time step, i: unit index, g: group index, nu: number of units in group, c: connection index, rc: return code
    Group *gto;
    Connections *connection;

    ng = net->numGroups;
    nt = net->runfor;
    nc = net->numConnections;

    /* zero squiggles */
    for (t = 0; t < nt; t++)
        for (g = 0; g < ng; g++)
        {

            gto = net->groups[g];
            nu = gto->numUnits;
            for (i = 0; i < nu; i++)
            {
                gto->dedx[t][i] = 0.0; // initialize dedx to zero
            }
        }

    for (t = nt - 1; t >= 0; t--) // BPTT
    {
        rc = (*net->preComputeGradientsMethod)(net, ex, t);
        if (rc == 0)
            continue;

        /* First we compute the dE/dY (e(k) from Williams
	 and Peng) */
        for (g = 0; g < ng; g++)
        {
            gto = net->groups[g];
            rc = (*gto->preTargetSetMethod)(gto, ex, t);
            if (rc)
                apply_example_targets(ex, gto, t);
            (*gto->postTargetSetMethod)(gto, ex, t);
            rc = (*gto->preComputeDeDxMethod)(gto, ex, t);
            if (rc)
            {
                nu = gto->numUnits;
                for (i = 0; i < nu; i++) // unit loop
                {
                    bptt_compute_dedx(net, gto, ex, t, i); // compute dE/dX
                }
            }
            (*gto->postComputeDeDxMethod)(gto, ex, t);
        }
        /* all dedx values now computed. */

        /* now we propagate back other e(k)s if not
	 initial time tick.  this is the big time sink. */
        if (t > 0)
        {
            for (c = 0; c < nc; c++)
            {
                connection = net->connections[c];
                bptt_backprop_error(net, connection, t);
            }
        }
        (*net->postComputeGradientsMethod)(net, ex, t);
    }
    return 1;
}

// 3. Compute the dE/dX

void bptt_compute_dedx(Net *net, Group *gto, Example *ex, int t, int i)
{
    Real local_err = 0.0, deriv, d = 0.0, out, target;

    /* note: here, dedx is a holder for dE/dy,
     not dE/dx */
    local_err = 0.0;
    if (VAL(gto->exampleData[i])) // #define VAL(x) (x > (-500.0))??? Maybe validate the label (target)?
    {
        out = gto->outputs[t][i];
        target = gto->exampleData[i];
        if (gto->targetNoise > 0.0)
        {
            target += gto->targetNoise * get_gaussian();
        }
        local_err = unit_error(gto, ex, i, out, target) * 2.0;
        d = unit_ce_distance(gto, ex, i, out, target); // Cross entropy distance
    }

    if (gto->errorRamp == RAMP_ERROR) // Many example default to 1...  wtf is this... No mention in HS04, nor Harm's thesis...
    {
        local_err *= (float)t / ((float)net->runfor - 1); //  local error =  local error * t / (nt - 1); JACKPOT: TICK 1 ISSUE MAYBE!!!
        d *= (float)t / ((float)net->runfor - 1);         // d = d * t / (nt - 1)
    }

    deriv = bptt_unit_derivative(gto, i, t); // unit derivative
    if (gto->errorComputation == SUM_SQUARED_ERROR)
    {
        gto->dedx[t][i] += local_err;
        gto->dedx[t][i] *= deriv;
    }
    else if (gto->errorComputation == CROSS_ENTROPY_ERROR) // BCE here...
    {
        gto->dedx[t][i] *= deriv; // dE/dx = dE/dy * dy/dx??? Chain rule??? seems not... WTF...   see Peng Williams, 1990 equation 19. not really relevant...
        gto->dedx[t][i] += d;
    }
    else
        Error0("Unknown errorComputation type");
}

// 4. Unit derivative

Real bptt_unit_derivative(Group *group, int unit, int tick)
{
    Real d;
    if (group->activationType == LOGISTIC_ACTIVATION)
        d = (sigmoid_derivative(group->outputs[tick][unit], // d_sigmoid
                                group->temperature));
    else if (group->activationType == TANH_ACTIVATION)
        d = (tanh_derivative(group->outputs[tick][unit],
                             group->temperature));

    else
    {
        Choke0("Group %s does not have legal activationType", group->name);
        exit(-1);
        return 0; /* just so stupid compilers don't complain */
    }
    d += group->primeOffset;
    return d;
}

// 4.5. Sigmoid derivative

double sigmoid_derivative(Real y, Real temp)
{
    double v;
    v = (double)temp * (double)y;

    v = CLIP(v, LOGISTIC_MIN, LOGISTIC_MAX);
    return ((double)temp) * ((double)y * (1.0 - (double)y));
}

// 5. Cross entropy distance

Real unit_ce_distance(Group *g, Example *ex, int unit, Real out, Real target)
{
    Real d;
    int gnum;

    gnum = g->index;

    if (fabs(out - target) < g->errorRadius) // Absolute error < ZERO_ERROR_RADIUS, then return 0
        return 0.0;

    if (g->activationType == TANH_ACTIVATION)
    {
        out = (out / 2.0) + 0.5;       /* cast it to 0 and 1 */
        target = (target / 2.0) + 0.5; /* cast it to 0 and 1 */
    }

    d = out - target; // Otherwise, return the plain distance, //basically works the same as ZER replace

    d = scale_error(g, ex, unit, d);

    return d;
}

// 6. Backpropagate error

void bptt_backprop_error(Net *net, Connections *c, int t)
{
    Group *gto, *gfrom;
    Real *w;
    Real mysquig, *outs, *gradient, *squig;
    int nu, nfrom, j, i, d;

    gto = c->to;
    gfrom = c->from;
    nu = gto->numUnits;
    nfrom = gfrom->numUnits;
    /* backprop squiggles and tweek gradients */

    if (gfrom->numIncoming == 0) // No incoming connections...
    {
        for (i = 0; i < nu; i++)
        {
            mysquig = gto->dedx[t][i] * c->scaleGradients;
            d = t - gto->delays[i];
            /* d is the time on the 'from' unit which affected
	     our 'to' unit. default is 1 */
            if (d >= 0)
            {
                outs = gfrom->outputs[d];
                gradient = c->gradients[i];
                for (j = 0; j < nfrom; j++)
                {
                    *gradient++ += mysquig * *outs++;
                }
            }
        }
    }
    else
    {
        for (i = 0; i < nu; i++)
        {
            /* d is the time on the 'from' unit which affected
	     our 'to' unit. default is 1 */
            d = t - gto->delays[i];
            if (d >= 0)
            {
                outs = gfrom->outputs[d];
                w = c->weights[i];
                mysquig = gto->dedx[t][i] * c->scaleGradients; // What is scaleGradients??? default is 1.0 in net.c
                squig = gfrom->dedx[d];
                gradient = c->gradients[i];
                for (j = 0; j < nfrom; j++)
                {
                    *squig++ += // *x++: pointer x incremented by 1
                        *w++ * mysquig;
                    *gradient++ += mysquig * *outs++;
                }
            }
        }
    }
}

// 7. Cross entropy error
Real ce_error(Group *g, Example *ex, int unit, Real out, Real target)
{
    Real local_error;
    int gnum;

    gnum = g->index;

    if (fabs(out - target) < g->errorRadius)   // Absolute error < ZERO_ERROR_RADIUS, then return 0
        return 0.0;

    if (g->activationType == TANH_ACTIVATION)
    {
        out = (out / 2.0) + 0.5;       /* cast it to 0 and 1 */
        target = (target / 2.0) + 0.5; /* cast it to 0 and 1 */
    }

    out = CLIP(out, LOGISTIC_MIN, LOGISTIC_MAX);


    // Bring max error to 1-ZERO_ERROR_RADIUS at both positive and negative side
    if (fabs(target) < 0.001) /* close enought to zero */
    {
        /* can't be higher than 1-errorRadius */
        out = CLIP(out, LOGISTIC_MIN, (1.0 - g->errorRadius));
        local_error = -log(1.0 - out);
    }
    else if (fabs(target) > 0.999) /* close enough to 1 */
    {
        /* can't be lower than errorRadius */
        out = CLIP(out, g->errorRadius, LOGISTIC_MAX);
        local_error = -log(out);
    }
    else
    {
        local_error = (log((target) / (out)) * (target) +
                       log((1.0 - (target)) /
                           (1.0 - (out))) *
                           (1.0 - (target)));
    }

    local_error = scale_error(g, ex, unit, local_error);
    return local_error;
}