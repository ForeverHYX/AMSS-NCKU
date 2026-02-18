

! Because of overlap determination, source region is always larger than target
! region

#include "macrodef.fh"

!--------------------------------------------------------------------------
!
! Prepare the data on coarse level for prolong
! valid for all finite difference order
!--------------------------------------------------------------------------

  subroutine prolongcopy3(wei,llbc,uubc,extc,func,&
                      llbf,uubf,exto,funo,&
                      llbp,uubp,SoA,Symmetry)
  implicit none

!~~~~~~> input arguments
  integer,intent(in) :: wei
!                                       coarse    fine       coarse
  real*8,dimension(3),   intent(in) :: llbc,uubc,llbf,uubf,llbp,uubp
  integer,dimension(3),  intent(in) :: extc,exto
  real*8, dimension(extc(1),extc(2),extc(3)),intent(in)   :: func
! both bounds ghost_width
  real*8, dimension(exto(1)+2*ghost_width,exto(2)+2*ghost_width,exto(3)+2*ghost_width),intent(out):: funo
  real*8, dimension(1:3), intent(in) :: SoA
  integer,intent(in)::Symmetry

!~~~~~~> local variables

  real*8,dimension(1-ghost_width:extc(1),1-ghost_width:extc(2),1-ghost_width:extc(3)) :: fh
  real*8, dimension(1:3) :: base
  integer,dimension(3) :: lbc,ubc,lbf,ubf,lbp,ubp,lbpc,ubpc,cxI
  integer :: i,j,k

  integer::imini,imaxi,jmini,jmaxi,kmini,kmaxi
  integer::imino,imaxo,jmino,jmaxo,kmino,kmaxo

  real*8,dimension(3) :: CD,FD

  if(wei.ne.3)then
     write(*,*)"prolongrestrict.f90::prolongcopy3: this routine only surport 3 dimension"
     write(*,*)"dim = ",wei
     stop
  endif

! it's possible a iolated point for target but not for source
  CD = (uubc-llbc)/extc
  FD = CD/2

!take care the mismatch of the two segments of grid
  do i=1,3
     if(llbc(i) <= llbf(i))then
        base(i) = llbc(i)
     else
        j=idint((llbc(i)-llbf(i))/FD(i)+0.4)
        if(j/2*2 == j)then
           base(i) = llbf(i)
        else
           base(i) = llbf(i) - CD(i)/2
        endif
     endif
  enddo

!!! function idint:
!If A is of type REAL and |A| < 1, INT(A) equals 0. If |A| \geq 1, 
!then INT(A) equals the largest integer that does not exceed the range of A 
!and whose sign is the same as the sign of A.

    lbf = idint((llbf-base)/FD+0.4)+1
    ubf = idint((uubf-base)/FD+0.4)
    lbc = idint((llbc-base)/CD+0.4)+1
    ubc = idint((uubc-base)/CD+0.4)
    lbp = idint((llbp-base)/FD+0.4)+1
    lbpc = idint((llbp-base)/CD+0.4)+1
    ubp = idint((uubp-base)/FD+0.4)
    ubpc = idint((uubp-base)/CD+0.4)
!sanity check
!|---*--- ---*--- ---*--- ---*--- ---*--- ---*--- ---*--- ---*---| 
!|=======x===============x===============x===============x=======|
!        ^                                               ^
  imini=lbpc(1)-lbc(1) + 1 - ghost_width
  imaxi=ubpc(1)-lbc(1) + 1 + ghost_width
  jmini=lbpc(2)-lbc(2) + 1 - ghost_width
  jmaxi=ubpc(2)-lbc(2) + 1 + ghost_width
  kmini=lbpc(3)-lbc(3) + 1 - ghost_width
  kmaxi=ubpc(3)-lbc(3) + 1 + ghost_width

  cxI(1) = imaxi-imini+1
  cxI(2) = jmaxi-jmini+1
  cxI(3) = kmaxi-kmini+1
  if(any(cxI.ne.exto+2*ghost_width).or. &
     imaxi.gt.extc(1)+1.or.jmaxi.gt.extc(2)+1.or.kmaxi.gt.extc(3)+1)then
          write(*,*)"error in prolongationcopy3 for"
          if(any(cxI.ne.exto+2*ghost_width))then
             write(*,*) cxI,exto+2*ghost_width
             return
          endif
          write(*,*)"from"
          write(*,*)llbc,uubc
          write(*,*)lbc,ubc
          write(*,*)"to"
          write(*,*)llbf,uubf
          write(*,*)lbf,ubf
          write(*,*)"want"
          write(*,*)llbp,uubp
          write(*,*)lbp,ubp,lbpc,ubpc
          if(imini.lt.1) write(*,*)"imini = ",imini
          if(jmini.lt.1) write(*,*)"jmini = ",jmini       
          if(kmini.lt.1) write(*,*)"kmini = ",kmini     
          if(imaxi.gt.extc(1)) write(*,*)"imaxi = ",imaxi,"extc(1) = ",extc(1) 
          if(jmaxi.gt.extc(2)) write(*,*)"jmaxi = ",jmaxi,"extc(2) = ",extc(2) 
          if(kmaxi.gt.extc(3)) write(*,*)"kmaxi = ",kmaxi,"extc(3) = ",extc(3) 
          return
  endif

! because some point needs 2*ghost_width
! while   some point needs 2*ghost_width-1
! so we use 0 to fill empty points
  if(imini < 1.or.jmini < 1.or.kmini < 1)then
    if(imini<1.and.dabs(llbp(1))>CD(1)) write(*,*)"prolongcopy3 warning: ",llbp(1)
    if(jmini<1.and.dabs(llbp(2))>CD(2)) write(*,*)"prolongcopy3 warning: ",llbp(2)
    if(kmini<1.and.dabs(llbp(3))>CD(3)) write(*,*)"prolongcopy3 warning: ",llbp(3)
    call symmetry_bd(ghost_width,extc,func,fh,SoA)
    if(imaxi<=extc(1).and.jmaxi<=extc(2).and.kmaxi<=extc(3))then
      funo = fh(imini:imaxi,jmini:jmaxi,kmini:kmaxi)
    else
      funo = 0.d0
      cxI = 0
      if(imaxi>extc(1))then
        cxI(1) = 1
        imaxi = extc(1)
      endif
      if(jmaxi>extc(2))then
        cxI(2) = 1
        jmaxi = extc(2)
      endif
      if(kmaxi>extc(3))then
        cxI(3) = 1
        kmaxi = extc(3)
      endif
      funo(1:exto(1)+2*ghost_width-cxI(1), &
           1:exto(2)+2*ghost_width-cxI(2), &
           1:exto(3)+2*ghost_width-cxI(3)) = fh(imini:imaxi,jmini:jmaxi,kmini:kmaxi)
    endif
  else
    if(imaxi<=extc(1).and.jmaxi<=extc(2).and.kmaxi<=extc(3))then
      funo = func(imini:imaxi,jmini:jmaxi,kmini:kmaxi)
    else
      funo = 0.d0
      cxI = 0
      if(imaxi>extc(1))then
        cxI(1) = 1
        imaxi = extc(1)
      endif
      if(jmaxi>extc(2))then
        cxI(2) = 1
        jmaxi = extc(2)
      endif
      if(kmaxi>extc(3))then
        cxI(3) = 1
        kmaxi = extc(3)
      endif
      funo(1:exto(1)+2*ghost_width-cxI(1), &
           1:exto(2)+2*ghost_width-cxI(2), &
           1:exto(3)+2*ghost_width-cxI(3)) = func(imini:imaxi,jmini:jmaxi,kmini:kmaxi)
    endif
  endif

  return

  end subroutine prolongcopy3
!=================================================================================================
#define MIX 0
!--------------------------------------------------------------------------
!
! Prolong data throug mix data of fine and coarse levels
!--------------------------------------------------------------------------

  subroutine prolongmix3(wei,llbf,uubf,extf,funf,&
                      llbc,uubc,exti,funi,&
                      llbp,uubp,SoA,Symmetry, &
                      illb,iuub)
  implicit none

!~~~~~~> input arguments
  integer,intent(in) :: wei
!                                       coarse      fine     coarse   fine (real inner points)
  real*8,dimension(3),   intent(in) :: llbc,uubc,llbf,uubf,llbp,uubp,illb,iuub
  integer,dimension(3),  intent(in) :: exti,extf
  real*8, dimension(extf(1),extf(2),extf(3)),intent(inout)   :: funf
! lower bound ghost_width; upper bound ghost_width-1  
  real*8, dimension(exti(1)+2*ghost_width,exti(2)+2*ghost_width,exti(3)+2*ghost_width),intent(in):: funi
  real*8, dimension(1:3), intent(in) :: SoA
  integer,intent(in)::Symmetry

!~~~~~~> local variables

  real*8, dimension(1:3) :: base
  integer,dimension(3) :: lbc,ubc,lbf,ubf,lbp,ubp,lbpc,ubpc,ilb,iub
  integer :: i,j,k,n,ii,jj,kk

  integer::imino,imaxo,jmino,jmaxo,kmino,kmaxo

  real*8,dimension(3) :: CD,FD
  integer,dimension(3) :: cxI,cxB,cxT,fg

  integer, parameter :: NO_SYMM = 0, EQ_SYMM = 1, OCTANT = 2

  real*8,dimension(2*ghost_width,2*ghost_width,2*ghost_width) :: ya
  real*8,dimension(2*ghost_width) :: X,Y,Z
  real*8, dimension(2*ghost_width,2*ghost_width) :: tmp2
  real*8, dimension(2*ghost_width) :: tmp1
  real*8 :: ddy
  real*8,dimension(3) :: ccp
  real*8, parameter :: C1=7.7d1/8.192d3,C2=-6.93d2/8.192d3,C3=3.465d3/4.096d3
  real*8, parameter :: C6=6.3d1/8.192d3,C5=-4.95d2/8.192d3,C4=1.155d3/4.096d3


  if(wei.ne.3)then
     write(*,*)"prolongrestrict_cell.f90::prolongmix3: this routine only surport 3 dimension"
     write(*,*)"dim = ",wei
     stop
  endif

! it's possible a iolated point for target but not for source
  FD = (uubf-llbf)/extf
  CD = FD*2.d0

!take care the mismatch of the two segments of grid
  do i=1,3
     if(llbc(i) <= llbf(i))then
        base(i) = llbc(i)
     else
        j=idint((llbc(i)-llbf(i))/FD(i)+0.4)
        if(j/2*2 == j)then
           base(i) = llbf(i)
        else
           base(i) = llbf(i) - CD(i)/2
        endif
     endif
  enddo

!!! function idint:
!If A is of type REAL and |A| < 1, INT(A) equals 0. If |A| \geq 1, 
!then INT(A) equals the largest integer that does not exceed the range of A 
!and whose sign is the same as the sign of A.

    lbf = idint((llbf-base)/FD+0.4)+1
    ubf = idint((uubf-base)/FD+0.4)
    lbc = idint((llbc-base)/CD+0.4)+1
    ubc = idint((uubc-base)/CD+0.4)
    lbp = idint((llbp-base)/FD+0.4)+1
    lbpc = idint((llbp-base)/CD+0.4)+1
    ubp = idint((uubp-base)/FD+0.4)
    ubpc = idint((uubp-base)/CD+0.4)
    ilb = idint((illb-base)/FD+0.4)+1
    iub = idint((iuub-base)/FD+0.4)
!sanity check
  imino=lbp(1)-lbf(1) + 1
  imaxo=ubp(1)-lbf(1) + 1
  jmino=lbp(2)-lbf(2) + 1
  jmaxo=ubp(2)-lbf(2) + 1
  kmino=lbp(3)-lbf(3) + 1
  kmaxo=ubp(3)-lbf(3) + 1

!sanity check
!|---*--- ---*--- ---*--- ---*--- ---*--- ---*--- ---*--- ---*---| 
!|=======x===============x===============x===============x=======|
!        ^                                               ^
! ghost_width for both sides
  lbpc = lbpc - ghost_width
  ubpc = ubpc + ghost_width
! index for real inner points  
  ilb = ilb - lbf+1
  iub = iub - lbf+1

  if(imino.lt.1.or.jmino.lt.1.or.kmino.lt.1.or.&
     imaxo.gt.extf(1).or.jmaxo.gt.extf(2).or.kmaxo.gt.extf(3))then
          write(*,*)"error in prolongmix3 for"
          write(*,*)"from"
          write(*,*)llbc,uubc
          write(*,*)lbc,ubc
          write(*,*)"to"
          write(*,*)llbf,uubf
          write(*,*)lbf,ubf
          write(*,*)base,FD
          write(*,*)"want"
          write(*,*)llbp,uubp
          write(*,*)lbp,ubp
          if(imino.lt.1) write(*,*)"imino = ",imino
          if(jmino.lt.1) write(*,*)"jmino = ",jmino       
          if(kmino.lt.1) write(*,*)"kmino = ",kmino
          if(imaxo.gt.extf(1)) write(*,*)"imaxo = ",imaxo,"extf(1) = ",extf(1) 
          if(jmaxo.gt.extf(2)) write(*,*)"jmaxo = ",jmaxo,"extf(2) = ",extf(2) 
          if(kmaxo.gt.extf(3)) write(*,*)"kmaxo = ",kmaxo,"extf(3) = ",extf(3) 
          return
  endif

  do k=kmino,kmaxo
  do j=jmino,jmaxo
  do i=imino,imaxo
       cxI(1) = i
       cxI(2) = j
       cxI(3) = k

       ccp = llbf+(cxI-0.5d0)*FD

! change to coarse level reference
!|---*--- ---*--- ---*--- ---*--- ---*--- ---*--- ---*--- ---*---| 
!|=======x===============x===============x===============x=======|
       cxI = (cxI+lbf-1)/2
! change to array index      
       cxI = cxI - lbpc + 1

       ya = funi(cxI(1)-ghost_width+1:cxI(1)+ghost_width,cxI(2)-ghost_width+1:cxI(2)+ghost_width,cxI(3)-ghost_width+1:cxI(3)+ghost_width)

       fg = 0
       where((illb.lt.ccp).and.(iuub.gt.ccp)) fg = 1

       if(sum(fg).eq.3)then
           write(*,*)"1 error in in prolongmix3:"
           write(*,*)ccp,illb,iuub
           stop
       endif

! fix the wanted point at (0,0,0), set FD = 1       
       ii=i+lbf(1)-1
       jj=j+lbf(2)-1
       kk=k+lbf(3)-1

       if(sum(fg).eq.2)then

       cxI(1) = i
       cxI(2) = j
       cxI(3) = k

!!!! set X       
       if(ii/2*2==ii)then
!            v
!|---*--- ---*--- ---*--- ---*--- ---*--- ---*--- ---*--- ---*---| 
!|=======x===============x===============x===============x=======|
         do n=1,ghost_width
           X(ghost_width-n+1) = -0.5d0-(n-1)*2
           X(ghost_width+n  ) =  1.5d0+(n-1)*2
         enddo
         if(cxI(1).gt.iub(1))then
            cxB(1) = iub(1)-ghost_width+1+(cxI(1)-iub(1)+1-MIX)/2
            cxT(1) = iub(1)
         elseif(cxI(1).lt.ilb(1))then
            cxB(1) = ilb(1)
            cxT(1) = ilb(1)+ghost_width-1-(ilb(1)-cxI(1)-MIX)/2
         elseif(fg(1).eq.0)then
           write(*,*)"2 error in in prolongmix3:"
           write(*,*)ccp(1),illb(1),iuub(1)
           stop
         endif
       else
!                    v
!|---*--- ---*--- ---*--- ---*--- ---*--- ---*--- ---*--- ---*---| 
!|=======x===============x===============x===============x=======|
         do n=1,ghost_width
           X(ghost_width-n+1) = -1.5d0-(n-1)*2
           X(ghost_width+n  ) =  0.5d0+(n-1)*2
         enddo
         if(cxI(1).gt.iub(1))then
            cxB(1) = iub(1)-ghost_width+1+(cxI(1)-iub(1)-MIX)/2
            cxT(1) = iub(1)
         elseif(cxI(1).lt.ilb(1))then
            cxB(1) = ilb(1)
            cxT(1) = ilb(1)+ghost_width-1-(ilb(1)-cxI(1)+1-MIX)/2
         elseif(fg(1).eq.0)then
           write(*,*)"3 error in in prolongmix3:"
           write(*,*)ccp(1),illb(1),iuub(1)
           stop
         endif
       endif

!!!! set Y       
       if(jj/2*2==jj)then
         do n=1,ghost_width
           Y(ghost_width-n+1) = -0.5d0-(n-1)*2
           Y(ghost_width+n  ) =  1.5d0+(n-1)*2
         enddo
         if(cxI(2).gt.iub(2))then
            cxB(2) = iub(2)-ghost_width+1+(cxI(2)-iub(2)+1-MIX)/2
            cxT(2) = iub(2)
         elseif(cxI(2).lt.ilb(2))then
            cxB(2) = ilb(2)
            cxT(2) = ilb(2)+ghost_width-1-(ilb(2)-cxI(2)-MIX)/2
         elseif(fg(2).eq.0)then
           write(*,*)"4 error in in prolongmix3:"
           write(*,*)ccp(2),illb(2),iuub(2)
           stop
         endif
       else
         do n=1,ghost_width
           Y(ghost_width-n+1) = -1.5d0-(n-1)*2
           Y(ghost_width+n  ) =  0.5d0+(n-1)*2
         enddo
         if(cxI(2).gt.iub(2))then
            cxB(2) = iub(2)-ghost_width+1+(cxI(2)-iub(2)-MIX)/2
            cxT(2) = iub(2)
         elseif(cxI(2).lt.ilb(2))then
            cxB(2) = ilb(2)
            cxT(2) = ilb(2)+ghost_width-1-(ilb(2)-cxI(2)+1-MIX)/2
         elseif(fg(2).eq.0)then
           write(*,*)"5 error in in prolongmix3:"
           write(*,*)ccp(2),illb(2),iuub(2)
           stop
         endif
       endif

!!!! set Z      
       if(kk/2*2==kk)then
         do n=1,ghost_width
           Z(ghost_width-n+1) = -0.5d0-(n-1)*2
           Z(ghost_width+n  ) =  1.5d0+(n-1)*2
         enddo
         if(cxI(3).gt.iub(3))then
            cxB(3) = iub(3)-ghost_width+1+(cxI(3)-iub(3)+1-MIX)/2
            cxT(3) = iub(3)
         elseif(cxI(3).lt.ilb(3))then
            cxB(3) = ilb(3)
            cxT(3) = ilb(3)+ghost_width-1-(ilb(3)-cxI(3)-MIX)/2
         elseif(fg(3).eq.0)then
           write(*,*)"6 error in in prolongmix3:"
           write(*,*)ccp(3),illb(3),iuub(3)
           stop
         endif
       else
         do n=1,ghost_width
           Z(ghost_width-n+1) = -1.5d0-(n-1)*2
           Z(ghost_width+n  ) =  0.5d0+(n-1)*2
         enddo
         if(cxI(3).gt.iub(3))then
            cxB(3) = iub(3)-ghost_width+1+(cxI(3)-iub(3)-MIX)/2
            cxT(3) = iub(3)
         elseif(cxI(3).lt.ilb(3))then
            cxB(3) = ilb(3)
            cxT(3) = ilb(3)+ghost_width-1-(ilb(3)-cxI(3)+1-MIX)/2
         elseif(fg(3).eq.0)then
           write(*,*)"7 error in in prolongmix3:"
           write(*,*)ccp(3),illb(3),iuub(3)
           stop
         endif
       endif

       endif
! X, Y, and Z are possiblly not in order, I assume polint does not
! require this order 
! because of the mismatch of points for fine level and coarse level 
! we have to deal in this way

! for x direction
       if(sum(fg).eq.2.and.fg(1) .eq. 0.and. &
         (((cxI(1).gt.iub(1)).and.(ghost_width-cxI(1)+cxB(1)+1.gt.0)).or. &
           (cxI(1).lt.ilb(1)).and.(ghost_width-cxI(1)+cxT(1).le.2*ghost_width)))then

         if(jj/2*2==jj)then
           if(kk/2*2==kk)then
             tmp2= C1*ya(:,:,1)+C2*ya(:,:,2)+C3*ya(:,:,3)+C4*ya(:,:,4)+C5*ya(:,:,5)+C6*ya(:,:,6)
             tmp1= C1*tmp2(:,1)+C2*tmp2(:,2)+C3*tmp2(:,3)+C4*tmp2(:,4)+C5*tmp2(:,5)+C6*tmp2(:,6)
           else
             tmp2= C6*ya(:,:,1)+C5*ya(:,:,2)+C4*ya(:,:,3)+C3*ya(:,:,4)+C2*ya(:,:,5)+C1*ya(:,:,6)
             tmp1= C1*tmp2(:,1)+C2*tmp2(:,2)+C3*tmp2(:,3)+C4*tmp2(:,4)+C5*tmp2(:,5)+C6*tmp2(:,6)
           endif
         else
           if(kk/2*2==kk)then
             tmp2= C1*ya(:,:,1)+C2*ya(:,:,2)+C3*ya(:,:,3)+C4*ya(:,:,4)+C5*ya(:,:,5)+C6*ya(:,:,6)
             tmp1= C6*tmp2(:,1)+C5*tmp2(:,2)+C4*tmp2(:,3)+C3*tmp2(:,4)+C2*tmp2(:,5)+C1*tmp2(:,6)
           else
             tmp2= C6*ya(:,:,1)+C5*ya(:,:,2)+C4*ya(:,:,3)+C3*ya(:,:,4)+C2*ya(:,:,5)+C1*ya(:,:,6)
             tmp1= C6*tmp2(:,1)+C5*tmp2(:,2)+C4*tmp2(:,3)+C3*tmp2(:,4)+C2*tmp2(:,5)+C1*tmp2(:,6)
           endif
         endif

         if(cxI(1).gt.iub(1))then
! consistent to coarse level, always X(ghost_width+1) = 0 for left
            do n=cxB(1),cxT(1)
              X(ghost_width-cxI(1)+n+1) = dble(n-cxI(1))
            enddo
            tmp1(ghost_width-cxI(1)+cxB(1)+1:ghost_width-cxI(1)+cxT(1)+1) = funf(cxB(1):cxT(1),j,k)
         elseif(cxI(1).lt.ilb(1))then
! consistent to coarse level, always X(ghost_width  ) = 0 for right
            do n=cxB(1),cxT(1)
              X(ghost_width-cxI(1)+n  ) = dble(n-cxI(1))
            enddo
            tmp1(ghost_width-cxI(1)+cxB(1)  :ghost_width-cxI(1)+cxT(1)  ) = funf(cxB(1):cxT(1),j,k)
         endif

         call polint(X,tmp1,0.d0,funf(i,j,k),ddy,2*ghost_width)

! for y direction
       elseif(sum(fg).eq.2.and.fg(2) .eq. 0.and. &
         (((cxI(2).gt.iub(2)).and.(ghost_width-cxI(2)+cxB(2)+1.gt.0)).or. &
           (cxI(2).lt.ilb(2)).and.(ghost_width-cxI(2)+cxT(2).le.2*ghost_width)))then

         if(ii/2*2==ii)then
           if(kk/2*2==kk)then
             tmp2= C1*ya(:,:,1)+C2*ya(:,:,2)+C3*ya(:,:,3)+C4*ya(:,:,4)+C5*ya(:,:,5)+C6*ya(:,:,6)
             tmp1= C1*tmp2(1,:)+C2*tmp2(2,:)+C3*tmp2(3,:)+C4*tmp2(4,:)+C5*tmp2(5,:)+C6*tmp2(6,:)
           else
             tmp2= C6*ya(:,:,1)+C5*ya(:,:,2)+C4*ya(:,:,3)+C3*ya(:,:,4)+C2*ya(:,:,5)+C1*ya(:,:,6)
             tmp1= C1*tmp2(1,:)+C2*tmp2(2,:)+C3*tmp2(3,:)+C4*tmp2(4,:)+C5*tmp2(5,:)+C6*tmp2(6,:)
           endif
         else
           if(kk/2*2==kk)then
             tmp2= C1*ya(:,:,1)+C2*ya(:,:,2)+C3*ya(:,:,3)+C4*ya(:,:,4)+C5*ya(:,:,5)+C6*ya(:,:,6)
             tmp1= C6*tmp2(1,:)+C5*tmp2(2,:)+C4*tmp2(3,:)+C3*tmp2(4,:)+C2*tmp2(5,:)+C1*tmp2(6,:)
           else
             tmp2= C6*ya(:,:,1)+C5*ya(:,:,2)+C4*ya(:,:,3)+C3*ya(:,:,4)+C2*ya(:,:,5)+C1*ya(:,:,6)
             tmp1= C6*tmp2(1,:)+C5*tmp2(2,:)+C4*tmp2(3,:)+C3*tmp2(4,:)+C2*tmp2(5,:)+C1*tmp2(6,:)
           endif
         endif     
         if(cxI(2).gt.iub(2))then
! consistent to coarse level, always Y(ghost_width+1) = 0 for left
            do n=cxB(2),cxT(2)
              Y(ghost_width-cxI(2)+n+1) = dble(n-cxI(2))
            enddo
            tmp1(ghost_width-cxI(2)+cxB(2)+1:ghost_width-cxI(2)+cxT(2)+1) = funf(i,cxB(2):cxT(2),k)
         elseif(cxI(2).lt.ilb(2))then
! consistent to coarse level, always Y(ghost_width  ) = 0 for right
            do n=cxB(2),cxT(2)
              Y(ghost_width-cxI(2)+n  ) = dble(n-cxI(2)) 
            enddo
            tmp1(ghost_width-cxI(2)+cxB(2)  :ghost_width-cxI(2)+cxT(2)  ) = funf(i,cxB(2):cxT(2),k)
         endif

         call polint(Y,tmp1,0.d0,funf(i,j,k),ddy,2*ghost_width)

! for z direction
       elseif(sum(fg).eq.2.and.fg(3) .eq. 0.and. &
         (((cxI(3).gt.iub(3)).and.(ghost_width-cxI(3)+cxB(3)+1.gt.0)).or. &
           (cxI(3).lt.ilb(3)).and.(ghost_width-cxI(3)+cxT(3).le.2*ghost_width)))then

         if(jj/2*2==jj)then
           if(ii/2*2==ii)then
             tmp2= C1*ya(1,:,:)+C2*ya(2,:,:)+C3*ya(3,:,:)+C4*ya(4,:,:)+C5*ya(5,:,:)+C6*ya(6,:,:)
             tmp1= C1*tmp2(1,:)+C2*tmp2(2,:)+C3*tmp2(3,:)+C4*tmp2(4,:)+C5*tmp2(5,:)+C6*tmp2(6,:)
           else
             tmp2= C6*ya(1,:,:)+C5*ya(2,:,:)+C4*ya(3,:,:)+C3*ya(4,:,:)+C2*ya(5,:,:)+C1*ya(6,:,:)
             tmp1= C1*tmp2(1,:)+C2*tmp2(2,:)+C3*tmp2(3,:)+C4*tmp2(4,:)+C5*tmp2(5,:)+C6*tmp2(6,:)
           endif
         else
           if(ii/2*2==ii)then
             tmp2= C1*ya(1,:,:)+C2*ya(2,:,:)+C3*ya(3,:,:)+C4*ya(4,:,:)+C5*ya(5,:,:)+C6*ya(6,:,:)
             tmp1= C6*tmp2(1,:)+C5*tmp2(2,:)+C4*tmp2(3,:)+C3*tmp2(4,:)+C2*tmp2(5,:)+C1*tmp2(6,:)
           else
             tmp2= C6*ya(1,:,:)+C5*ya(2,:,:)+C4*ya(3,:,:)+C3*ya(4,:,:)+C2*ya(5,:,:)+C1*ya(6,:,:)
             tmp1= C6*tmp2(1,:)+C5*tmp2(2,:)+C4*tmp2(3,:)+C3*tmp2(4,:)+C2*tmp2(5,:)+C1*tmp2(6,:)
           endif
         endif

         if(cxI(3).gt.iub(3))then
! consistent to coarse level, always Z(ghost_width+1) = 0 for left
            do n=cxB(3),cxT(3)
              Z(ghost_width-cxI(3)+n+1) = dble(n-cxI(3))
            enddo
            tmp1(ghost_width-cxI(3)+cxB(3)+1:ghost_width-cxI(3)+cxT(3)+1) = funf(i,j,cxB(3):cxT(3))
         elseif(cxI(3).lt.ilb(3))then
! consistent to coarse level, always Z(ghost_width  ) = 0 for right
            do n=cxB(3),cxT(3)
              Z(ghost_width-cxI(3)+n  ) = dble(n-cxI(3)) 
            enddo
            tmp1(ghost_width-cxI(3)+cxB(3)  :ghost_width-cxI(3)+cxT(3)  ) = funf(i,j,cxB(3):cxT(3))
         endif

         call polint(Z,tmp1,0.d0,funf(i,j,k),ddy,2*ghost_width)
     
       else

       if(ii/2*2==ii)then
         if(jj/2*2==jj)then
           if(kk/2*2==kk)then
             tmp2= C1*ya(:,:,1)+C2*ya(:,:,2)+C3*ya(:,:,3)+C4*ya(:,:,4)+C5*ya(:,:,5)+C6*ya(:,:,6)
             tmp1= C1*tmp2(:,1)+C2*tmp2(:,2)+C3*tmp2(:,3)+C4*tmp2(:,4)+C5*tmp2(:,5)+C6*tmp2(:,6)
             funf(i,j,k)= C1*tmp1(1)+C2*tmp1(2)+C3*tmp1(3)+C4*tmp1(4)+C5*tmp1(5)+C6*tmp1(6)
           else
             tmp2= C6*ya(:,:,1)+C5*ya(:,:,2)+C4*ya(:,:,3)+C3*ya(:,:,4)+C2*ya(:,:,5)+C1*ya(:,:,6)
             tmp1= C1*tmp2(:,1)+C2*tmp2(:,2)+C3*tmp2(:,3)+C4*tmp2(:,4)+C5*tmp2(:,5)+C6*tmp2(:,6)
             funf(i,j,k)=  C1*tmp1(1)+C2*tmp1(2)+C3*tmp1(3)+C4*tmp1(4)+C5*tmp1(5)+C6*tmp1(6)
           endif
         else
           if(kk/2*2==kk)then
             tmp2= C1*ya(:,:,1)+C2*ya(:,:,2)+C3*ya(:,:,3)+C4*ya(:,:,4)+C5*ya(:,:,5)+C6*ya(:,:,6)
             tmp1= C6*tmp2(:,1)+C5*tmp2(:,2)+C4*tmp2(:,3)+C3*tmp2(:,4)+C2*tmp2(:,5)+C1*tmp2(:,6)
             funf(i,j,k)= C1*tmp1(1)+C2*tmp1(2)+C3*tmp1(3)+C4*tmp1(4)+C5*tmp1(5)+C6*tmp1(6)
           else
             tmp2= C6*ya(:,:,1)+C5*ya(:,:,2)+C4*ya(:,:,3)+C3*ya(:,:,4)+C2*ya(:,:,5)+C1*ya(:,:,6)
             tmp1= C6*tmp2(:,1)+C5*tmp2(:,2)+C4*tmp2(:,3)+C3*tmp2(:,4)+C2*tmp2(:,5)+C1*tmp2(:,6)
             funf(i,j,k)=  C1*tmp1(1)+C2*tmp1(2)+C3*tmp1(3)+C4*tmp1(4)+C5*tmp1(5)+C6*tmp1(6)
           endif
         endif
       else
         if(jj/2*2==jj)then
           if(kk/2*2==kk)then               
             tmp2= C1*ya(:,:,1)+C2*ya(:,:,2)+C3*ya(:,:,3)+C4*ya(:,:,4)+C5*ya(:,:,5)+C6*ya(:,:,6)
             tmp1= C1*tmp2(:,1)+C2*tmp2(:,2)+C3*tmp2(:,3)+C4*tmp2(:,4)+C5*tmp2(:,5)+C6*tmp2(:,6)
             funf(i,j,k)= C6*tmp1(1)+C5*tmp1(2)+C4*tmp1(3)+C3*tmp1(4)+C2*tmp1(5)+C1*tmp1(6)
           else
             tmp2= C6*ya(:,:,1)+C5*ya(:,:,2)+C4*ya(:,:,3)+C3*ya(:,:,4)+C2*ya(:,:,5)+C1*ya(:,:,6)
             tmp1= C1*tmp2(:,1)+C2*tmp2(:,2)+C3*tmp2(:,3)+C4*tmp2(:,4)+C5*tmp2(:,5)+C6*tmp2(:,6)
             funf(i,j,k)=  C6*tmp1(1)+C5*tmp1(2)+C4*tmp1(3)+C3*tmp1(4)+C2*tmp1(5)+C1*tmp1(6)
           endif
         else
           if(kk/2*2==kk)then
             tmp2= C1*ya(:,:,1)+C2*ya(:,:,2)+C3*ya(:,:,3)+C4*ya(:,:,4)+C5*ya(:,:,5)+C6*ya(:,:,6)
             tmp1= C6*tmp2(:,1)+C5*tmp2(:,2)+C4*tmp2(:,3)+C3*tmp2(:,4)+C2*tmp2(:,5)+C1*tmp2(:,6)
             funf(i,j,k)= C6*tmp1(1)+C5*tmp1(2)+C4*tmp1(3)+C3*tmp1(4)+C2*tmp1(5)+C1*tmp1(6)
           else
             tmp2= C6*ya(:,:,1)+C5*ya(:,:,2)+C4*ya(:,:,3)+C3*ya(:,:,4)+C2*ya(:,:,5)+C1*ya(:,:,6)
             tmp1= C6*tmp2(:,1)+C5*tmp2(:,2)+C4*tmp2(:,3)+C3*tmp2(:,4)+C2*tmp2(:,5)+C1*tmp2(:,6)
             funf(i,j,k)=  C6*tmp1(1)+C5*tmp1(2)+C4*tmp1(3)+C3*tmp1(4)+C2*tmp1(5)+C1*tmp1(6)
           endif
         endif
       endif  
       endif

  enddo
  enddo
  enddo

  return

  end subroutine prolongmix3
!///////////////////////////////////////////////////////////////////////////////////////////////
! for different finite differnce order
! fourth order code
!--------------------------------------------------------------------------
!
! Prolongation from coarser grids to finer grids
! 6 points, 5th order interpolation
! 1   2   3   4   5   6
! *---*---*---*---*---*
!          ^
! f=77/8192*f_1 - 693/8192*f_2 + 3465/4096*f_3 +
!   63/8192*f_6 - 495/8192*f_5 + 1155/4096*f_4
!--------------------------------------------------------------------------
  subroutine prolong3(wei,llbc,uubc,extc,func,&
                      llbf,uubf,extf,funf,&
                      llbp,uubp,SoA,Symmetry)
  implicit none

!~~~~~~> input arguments
  integer,intent(in) :: wei
!                                       coarse    fine       fine
  real*8,dimension(3),   intent(in) :: llbc,uubc,llbf,uubf,llbp,uubp
  integer,dimension(3),  intent(in) :: extc,extf
  real*8, dimension(extc(1),extc(2),extc(3)),intent(in)   :: func
  real*8, dimension(extf(1),extf(2),extf(3)),intent(inout):: funf
  real*8, dimension(1:3), intent(in) :: SoA
  integer,intent(in)::Symmetry

!~~~~~~> local variables

  real*8, dimension(1:3) :: base
  integer,dimension(3) :: lbc,ubc,lbf,ubf,lbp,ubp,lbpc,ubpc
! when if=1 -> ic=0, this is different to vertex center grid 
  real*8, dimension(-2:extc(1),-2:extc(2),-2:extc(3))   :: funcc
  integer,dimension(3) :: cxI
  integer :: i,j,k,ii,jj,kk
  real*8, dimension(6,6) :: tmp2
  real*8, dimension(6) :: tmp1

  real*8, parameter :: C1=7.7d1/8.192d3,C2=-6.93d2/8.192d3,C3=3.465d3/4.096d3
  real*8, parameter :: C6=6.3d1/8.192d3,C5=-4.95d2/8.192d3,C4=1.155d3/4.096d3

  integer::imini,imaxi,jmini,jmaxi,kmini,kmaxi
  integer::imino,imaxo,jmino,jmaxo,kmino,kmaxo

  real*8,dimension(3) :: CD,FD
  
  if(wei.ne.3)then
     write(*,*)"prolongrestrict.f90::prolong3: this routine only surport 3 dimension"
     write(*,*)"dim = ",wei
     stop
  endif

  CD = (uubc-llbc)/extc
  FD = (uubf-llbf)/extf

  if(any(dabs(CD-2*FD)>1.d-10))then
     write(*,*)"prolong:",CD,FD
     stop
  endif

!take care the mismatch of the two segments of grid
  do i=1,3
     if(llbc(i) <= llbf(i))then
        base(i) = llbc(i)
     else
        j=idint((llbc(i)-llbf(i))/FD(i)+0.4)
        if(j/2*2 == j)then
           base(i) = llbf(i)
        else
           base(i) = llbf(i) - CD(i)/2
        endif
     endif
  enddo

!!! function idint:
!If A is of type REAL and |A| < 1, INT(A) equals 0. If |A| \geq 1, 
!then INT(A) equals the largest integer that does not exceed the range of A 
!and whose sign is the same as the sign of A.

    lbf = idint((llbf-base)/FD+0.4)+1
    ubf = idint((uubf-base)/FD+0.4)
    lbc = idint((llbc-base)/CD+0.4)+1
    ubc = idint((uubc-base)/CD+0.4)
    lbp = idint((llbp-base)/FD+0.4)+1
    lbpc = idint((llbp-base)/CD+0.4)+1  ! this is wrong, but not essential
    ubp = idint((uubp-base)/FD+0.4)
    ubpc = idint((uubp-base)/CD+0.4)    ! this is wrong, but not essential

!sanity check
  imino=lbp(1)-lbf(1) + 1
  imaxo=ubp(1)-lbf(1) + 1
  jmino=lbp(2)-lbf(2) + 1
  jmaxo=ubp(2)-lbf(2) + 1
  kmino=lbp(3)-lbf(3) + 1
  kmaxo=ubp(3)-lbf(3) + 1

  imini=lbpc(1)-lbc(1) + 1
  imaxi=ubpc(1)-lbc(1) + 1
  jmini=lbpc(2)-lbc(2) + 1
  jmaxi=ubpc(2)-lbc(2) + 1
  kmini=lbpc(3)-lbc(3) + 1
  kmaxi=ubpc(3)-lbc(3) + 1

  if(imino.lt.1.or.jmino.lt.1.or.kmino.lt.1.or.&
     imini.lt.1.or.jmini.lt.1.or.kmini.lt.1.or.&
     imaxo.gt.extf(1).or.jmaxo.gt.extf(2).or.kmaxo.gt.extf(3).or.&
     imaxi.gt.extc(1)-2.or.jmaxi.gt.extc(2)-2.or.kmaxi.gt.extc(3)-2)then
          write(*,*)"error in prolongation for"
          write(*,*)"from"
          write(*,*)llbc,uubc
          write(*,*)lbc,ubc
          write(*,*)"to"
          write(*,*)llbf,uubf
          write(*,*)lbf,ubf
          write(*,*)"want"
          write(*,*)llbp,uubp
          write(*,*)lbp,ubp,lbpc,ubpc
          return
  endif

  call symmetry_bd(3,extc,func,funcc,SoA)
     
!~~~~~~> prolongation start...
  do k = kmino,kmaxo
   do j = jmino,jmaxo
    do i = imino,imaxo
       cxI(1) = i
       cxI(2) = j
       cxI(3) = k
! change to coarse level reference
!|---*--- ---*--- ---*--- ---*--- ---*--- ---*--- ---*--- ---*---| 
!|=======x===============x===============x===============x=======|
       cxI = (cxI+lbf-1)/2
! change to array index      
       cxI = cxI - lbc + 1

       if(any(cxI+3 > extc)) write(*,*)"error in prolong"
       ii=i+lbf(1)-1
       jj=j+lbf(2)-1
       kk=k+lbf(3)-1
       if(kk/2*2==kk)then
             tmp2= C1*funcc(cxI(1)-2:cxI(1)+3,cxI(2)-2:cxI(2)+3,cxI(3)-2)+&
                   C2*funcc(cxI(1)-2:cxI(1)+3,cxI(2)-2:cxI(2)+3,cxI(3)-1)+&
                   C3*funcc(cxI(1)-2:cxI(1)+3,cxI(2)-2:cxI(2)+3,cxI(3)  )+&
                   C4*funcc(cxI(1)-2:cxI(1)+3,cxI(2)-2:cxI(2)+3,cxI(3)+1)+&
                   C5*funcc(cxI(1)-2:cxI(1)+3,cxI(2)-2:cxI(2)+3,cxI(3)+2)+&
                   C6*funcc(cxI(1)-2:cxI(1)+3,cxI(2)-2:cxI(2)+3,cxI(3)+3)
       else
             tmp2= C6*funcc(cxI(1)-2:cxI(1)+3,cxI(2)-2:cxI(2)+3,cxI(3)-2)+&
                   C5*funcc(cxI(1)-2:cxI(1)+3,cxI(2)-2:cxI(2)+3,cxI(3)-1)+&
                   C4*funcc(cxI(1)-2:cxI(1)+3,cxI(2)-2:cxI(2)+3,cxI(3)  )+&
                   C3*funcc(cxI(1)-2:cxI(1)+3,cxI(2)-2:cxI(2)+3,cxI(3)+1)+&
                   C2*funcc(cxI(1)-2:cxI(1)+3,cxI(2)-2:cxI(2)+3,cxI(3)+2)+&
                   C1*funcc(cxI(1)-2:cxI(1)+3,cxI(2)-2:cxI(2)+3,cxI(3)+3)
       endif

       if(jj/2*2==jj)then
             tmp1= C1*tmp2(:,1)+C2*tmp2(:,2)+C3*tmp2(:,3)+C4*tmp2(:,4)+C5*tmp2(:,5)+C6*tmp2(:,6)
       else
             tmp1= C6*tmp2(:,1)+C5*tmp2(:,2)+C4*tmp2(:,3)+C3*tmp2(:,4)+C2*tmp2(:,5)+C1*tmp2(:,6)
       endif

       if(ii/2*2==ii)then
             funf(i,j,k)= C1*tmp1(1)+C2*tmp1(2)+C3*tmp1(3)+C4*tmp1(4)+C5*tmp1(5)+C6*tmp1(6)
       else
             funf(i,j,k)= C6*tmp1(1)+C5*tmp1(2)+C4*tmp1(3)+C3*tmp1(4)+C2*tmp1(5)+C1*tmp1(6)
       endif
    enddo
   enddo
  enddo

  return

  end subroutine prolong3
!--------------------------------------------------------------------------
!
! Restrict from finner grids to coarser grids ignore the boundary point
!
! 6 points, 5th order interpolation
! 1   2   3   4   5   6
! *---*---*---*---*---*
!           ^
! f=3/256*(f_1+f_6) - 25/256*(f_2+f_5) + 75/128*(f_3+f_4)
!--------------------------------------------------------------------------
  subroutine restrict3(wei,llbc,uubc,extc,func,&
                       llbf,uubf,extf,funf,&
                       llbr,uubr,SoA,Symmetry)
  implicit none

!~~~~~~> input arguments
  integer,intent(in)::wei
!                                       coarse    fine       coarse
  real*8,dimension(3),   intent(in) :: llbc,uubc,llbf,uubf,llbr,uubr
  integer,dimension(3),  intent(in) :: extc,extf
  real*8, dimension(extc(1),extc(2),extc(3)),intent(inout):: func
  real*8, dimension(extf(1),extf(2),extf(3)),intent(in):: funf
  real*8, dimension(1:3), intent(in) :: SoA
  integer,intent(in)::Symmetry

!~~~~~~> local variables

  real*8, dimension(1:3) :: base
  integer,dimension(3) :: lbc,ubc,lbf,ubf,lbr,ubr,lbrf,ubrf
  real*8, dimension(-1:extf(1),-1:extf(2),-1:extf(3)):: funff
  integer,dimension(3) :: cxI
  integer :: i,j,k
  real*8, dimension(6,6) :: tmp2
  real*8, dimension(6) :: tmp1
  real*8, parameter :: C1=3.d0/2.56d2,C2=-2.5d1/2.56d2,C3=7.5d1/1.28d2

  integer::imini,imaxi,jmini,jmaxi,kmini,kmaxi
  integer::imino,imaxo,jmino,jmaxo,kmino,kmaxo

  real*8,dimension(3) :: CD,FD
  
  if(wei.ne.3)then
     write(*,*)"prolongrestrict.f90::restrict3: this routine only surport 3 dimension"
     write(*,*)"dim = ",wei
     stop
  endif

  CD = (uubc-llbc)/extc
  FD = (uubf-llbf)/extf

  if(any(dabs(CD-2*FD)>1.d-10))then
     write(*,*)"restrict:",CD,FD
     stop
  endif
!take care the mismatch of the two segments of grid
  do i=1,3
     if(llbc(i) <= llbf(i))then
        base(i) = llbc(i)
     else
        j=idint((llbc(i)-llbf(i))/FD(i)+0.4)
        if(j/2*2 == j)then
           base(i) = llbf(i)
        else
           base(i) = llbf(i) - CD(i)/2
        endif
     endif
  enddo
!!! function idint:
!If A is of type REAL and |A| < 1, INT(A) equals 0. If |A| \geq 1, 
!then INT(A) equals the largest integer that does not exceed the range of A 
!and whose sign is the same as the sign of A.

! note say base = 0, llbf = 0, uubf = 2
! llbf->1 and uubf->2
    lbf = idint((llbf-base)/FD+0.4)+1
    ubf = idint((uubf-base)/FD+0.4)
    lbc = idint((llbc-base)/CD+0.4)+1
    ubc = idint((uubc-base)/CD+0.4)
    lbr = idint((llbr-base)/CD+0.4)+1
    lbrf = idint((llbr-base)/FD+0.4)+1 !this is wrong but not essential
    ubr = idint((uubr-base)/CD+0.4)
    ubrf = idint((uubr-base)/FD+0.4)   !this is wrong but not essential

!sanity check
  imino=lbr(1)-lbc(1) + 1
  imaxo=ubr(1)-lbc(1) + 1
  jmino=lbr(2)-lbc(2) + 1
  jmaxo=ubr(2)-lbc(2) + 1
  kmino=lbr(3)-lbc(3) + 1
  kmaxo=ubr(3)-lbc(3) + 1

  imini=lbrf(1)-lbf(1) + 1
  imaxi=ubrf(1)-lbf(1) + 1
  jmini=lbrf(2)-lbf(2) + 1
  jmaxi=ubrf(2)-lbf(2) + 1
  kmini=lbrf(3)-lbf(3) + 1
  kmaxi=ubrf(3)-lbf(3) + 1

  if(imino.lt.1.or.jmino.lt.1.or.kmino.lt.1.or.&
     imini.lt.1.or.jmini.lt.1.or.kmini.lt.1.or.&
     imaxo.gt.extc(1).or.jmaxo.gt.extc(2).or.kmaxo.gt.extc(3).or.&
     imaxi.gt.extf(1)-2.or.jmaxi.gt.extf(2)-2.or.kmaxi.gt.extf(3)-2)then
          write(*,*)"error in restrict for"
          write(*,*)"from"
          write(*,*)lbf,ubf
          write(*,*)"to"
          write(*,*)lbc,ubc
          write(*,*)"want"
          write(*,*)lbr,ubr,lbrf,ubrf
          write(*,*)"llbf = ",llbf
          write(*,*)"uubf = ",uubf
          write(*,*)"llbc = ",llbc
          write(*,*)"uubc = ",uubc
          write(*,*)"llbr = ",llbr
          write(*,*)"uubr = ",uubr
          write(*,*)"base = ",base
          stop
  endif

  call symmetry_bd(2,extf,funf,funff,SoA)

!~~~~~~> restriction start...
  do k = kmino,kmaxo
   do j = jmino,jmaxo
    do i = imino,imaxo

       cxI(1) = i
       cxI(2) = j
       cxI(3) = k
! change to fine level reference
!|---*--- ---*--- ---*--- ---*--- ---*--- ---*--- ---*--- ---*---| 
!|=======x===============x===============x===============x=======|
       cxI = 2*(cxI+lbc-1) - 1
! change to array index      
       cxI = cxI - lbf + 1

       if(any(cxI+3 > extf)) write(*,*)"error in restrict"
       tmp2= C1*(funff(cxI(1)-2:cxI(1)+3,cxI(2)-2:cxI(2)+3,cxI(3)-2)+funff(cxI(1)-2:cxI(1)+3,cxI(2)-2:cxI(2)+3,cxI(3)+3))&
            +C2*(funff(cxI(1)-2:cxI(1)+3,cxI(2)-2:cxI(2)+3,cxI(3)-1)+funff(cxI(1)-2:cxI(1)+3,cxI(2)-2:cxI(2)+3,cxI(3)+2))&
            +C3*(funff(cxI(1)-2:cxI(1)+3,cxI(2)-2:cxI(2)+3,cxI(3)  )+funff(cxI(1)-2:cxI(1)+3,cxI(2)-2:cxI(2)+3,cxI(3)+1))
       tmp1= C1*(tmp2(:,1)+tmp2(:,6))+C2*(tmp2(:,2)+tmp2(:,5))+C3*(tmp2(:,3)+tmp2(:,4))
       func(i,j,k)= C1*(tmp1(1)+tmp1(6))+C2*(tmp1(2)+tmp1(5))+C3*(tmp1(3)+tmp1(4))
    enddo
   enddo
  enddo
  
  return

  end subroutine restrict3